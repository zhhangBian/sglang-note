//! Behavioral tests at the pure API boundary, using executable prefixes.
use sgl_kv_orchestrator::{
    models::{Compatibility, SessionIdentity},
    session_index::*,
    target_resolver::*,
};
use std::sync::Arc;

fn limits() -> SessionLimits {
    SessionLimits {
        max_sessions: 16,
        max_branches: 32,
        max_history_revisions: 64,
        max_requests: 64,
        max_content_nodes: 64,
        max_token_material_bytes: 4096,
        max_tombstones: 32,
        max_claims: 32,
        max_intents: 32,
        max_identifier_bytes: 128,
        max_components: 8,
    }
}
fn compatibility() -> Compatibility {
    Compatibility {
        namespace: "ns".into(),
        cache_salt: None,
        layout_id: "full:p4".into(),
        page_size: 4,
        components: vec!["full".into()],
        hash_scheme: sgl_kv_orchestrator::protocol::HASH_SCHEME.into(),
    }
}
fn key(digest: &str) -> ContentKey {
    ContentKey::new(compatibility(), digest.into()).unwrap()
}
fn session(name: &str) -> SessionIdentity {
    SessionIdentity {
        retirement_generation: None,
        caller_id: "caller".into(),
        session_id: name.into(),
        incarnation: "task-1".into(),
    }
}
fn request(id: &str, incarnation: &str) -> RequestIdentity {
    RequestIdentity {
        request_id: id.into(),
        incarnation: incarnation.into(),
    }
}
fn register(
    index: &mut SessionIndex,
    digest: &str,
    parent: Option<&ContentKey>,
    end: u64,
    tokens: Option<Vec<u32>>,
) -> ContentKey {
    let content = key(digest);
    index
        .register_content(ContentRegistration {
            key: content.clone(),
            parent: parent.cloned(),
            end_token: end,
            suffix_tokens: tokens,
        })
        .unwrap();
    content
}
fn association(
    task: &SessionIdentity,
    branch: &str,
    session_revision: u64,
    branch_revision: u64,
    content: &ContentKey,
    request: Option<RequestIdentity>,
) -> Association {
    Association {
        session: task.clone(),
        session_revision,
        branch_id: branch.into(),
        branch_revision,
        content: content.clone(),
        request,
        original_end: None,
        provenance: AssociationProvenance::RenderedRequest,
    }
}
fn bind(
    index: &mut SessionIndex,
    task: &SessionIdentity,
    branch: &str,
    session_revision: u64,
    branch_revision: u64,
    content: &ContentKey,
    request: Option<RequestIdentity>,
) -> Arc<BranchRevision> {
    let association = association(
        task,
        branch,
        session_revision,
        branch_revision,
        content,
        request,
    );
    index.associate(association).unwrap()
}
#[test]
fn branch_heads_advance_while_frozen_and_historical_requests_keep_their_scope() {
    let mut index = SessionIndex::new(limits());
    let task = session("chat");
    index.start_session(task.clone(), None).unwrap();
    let root = register(&mut index, "root", None, 4, Some(vec![1, 2, 3, 4]));
    let left = register(&mut index, "left", Some(&root), 8, Some(vec![5, 6, 7, 8]));
    let request_id = request("r", "sample-1");
    let old = bind(
        &mut index,
        &task,
        "main",
        1,
        1,
        &left,
        Some(request_id.clone()),
    );
    let right = register(
        &mut index,
        "right",
        Some(&root),
        8,
        Some(vec![9, 10, 11, 12]),
    );
    let sibling = bind(&mut index, &task, "side", 2, 1, &right, None);
    assert!(Arc::ptr_eq(
        old.content().parent().unwrap(),
        sibling.content().parent().unwrap()
    ));
    assert_eq!(index.usage().token_material_bytes, 3 * 4 * 4);

    let frozen = index.resolve(Selector::Session(task.clone()));
    let branches = frozen.branch_set();
    assert_eq!(branches.len(), 2);
    bind(
        &mut index,
        &task,
        "main",
        3,
        2,
        &root,
        Some(request("r", "sample-2")),
    );
    bind(&mut index, &task, "later", 4, 1, &root, None);
    assert_eq!(frozen.snapshot_revision(), Some(2));
    assert_eq!(frozen.branch_set(), branches);
    assert_eq!(
        index
            .resolve(Selector::Session(task.clone()))
            .branch_set()
            .len(),
        3
    );

    for selector in [
        Selector::Request {
            session: task.clone(),
            request: request_id,
        },
        Selector::Branch {
            session: task.clone(),
            branch_id: "main".into(),
            revision: Some(1),
        },
    ] {
        let historical = index.resolve(selector);
        let scope = &historical.known_scope()[0];
        assert!(Arc::ptr_eq(scope.snapshot().unwrap(), &old));
        assert_eq!(
            index.materialize(scope.scope()).unwrap().token_ids,
            Some((1..=8).collect())
        );
    }
    let latest = index.resolve(Selector::Branch {
        session: task,
        branch_id: "main".into(),
        revision: None,
    });
    assert_eq!(latest.known_scope()[0].scope().key(), &root);
}

#[test]
fn bounds_refuse_without_losing_active_metadata_and_reclaim_only_unreferenced_history() {
    let mut config = limits();
    config.max_history_revisions = 1;
    config.max_content_nodes = 1;
    config.max_branches = 1;
    let mut index = SessionIndex::new(config);
    let task = session("task");
    index.start_session(task.clone(), None).unwrap();
    let content = register(&mut index, "root", None, 4, Some(vec![1, 2, 3, 4]));
    let old = bind(
        &mut index,
        &task,
        "main",
        1,
        1,
        &content,
        Some(request("r", "1")),
    );
    let frozen = index.resolve(Selector::Session(task.clone()));
    assert_eq!(
        index
            .associate(association(&task, "main", 2, 2, &content, None))
            .unwrap_err(),
        IndexError::Capacity("history_revisions")
    );
    assert_eq!(
        index
            .associate(association(&task, "other", 2, 1, &content, None))
            .unwrap_err(),
        IndexError::Capacity("branches")
    );
    assert_eq!(
        index.register_content(ContentRegistration {
            key: key("new"),
            parent: Some(content.clone()),
            end_token: 8,
            suffix_tokens: Some(vec![5, 6, 7, 8])
        }),
        Err(IndexError::Capacity("content_nodes"))
    );
    assert_eq!(index.session_revision(&task), Some(1));
    index.forget_request(&task, &request("r", "1"), 2).unwrap();
    drop(old);
    assert_eq!(
        index
            .associate(association(&task, "main", 3, 2, &content, None))
            .unwrap_err(),
        IndexError::Capacity("history_revisions")
    );
    drop(frozen);
    bind(&mut index, &task, "main", 3, 2, &content, None);
    assert_eq!(index.usage().history_revisions, 1);
    assert!(index.branch(&task, "main", Some(1)).is_none());
}

#[test]
fn material_request_resource_and_tombstone_limits_fail_atomically() {
    let mut config = limits();
    config.max_sessions = 2;
    config.max_token_material_bytes = 16;
    config.max_requests = 1;
    config.max_claims = 1;
    config.max_intents = 1;
    config.max_tombstones = 2;
    let mut index = SessionIndex::new(config);
    let a = session("a");
    let b = session("b");
    index.start_session(a.clone(), None).unwrap();
    index.start_session(b.clone(), None).unwrap();
    assert_eq!(
        index.start_session(session("c"), None),
        Err(IndexError::Capacity("sessions"))
    );
    let root = register(&mut index, "root", None, 4, Some(vec![1, 2, 3, 4]));
    bind(&mut index, &a, "main", 1, 1, &root, Some(request("r", "1")));
    assert_eq!(
        index.register_content(ContentRegistration {
            key: key("leaf"),
            parent: Some(root.clone()),
            end_token: 8,
            suffix_tokens: Some(vec![5, 6, 7, 8])
        }),
        Err(IndexError::Capacity("token_material_bytes"))
    );
    assert_eq!(index.usage().content_nodes, 1);
    assert_eq!(
        index
            .associate(association(
                &b,
                "main",
                1,
                1,
                &root,
                Some(request("r", "2"))
            ))
            .unwrap_err(),
        IndexError::Capacity("requests")
    );
    assert_eq!(index.session_revision(&b), Some(0));
    index
        .track_resource(&a, ResourceKind::Claim, "c".into())
        .unwrap();
    index
        .track_resource(&a, ResourceKind::Intent, "i".into())
        .unwrap();
    assert_eq!(
        index.track_resource(&b, ResourceKind::Claim, "c".into()),
        Err(IndexError::Capacity("claims"))
    );
    assert_eq!(
        index.track_resource(&b, ResourceKind::Intent, "i".into()),
        Err(IndexError::Capacity("intents"))
    );
    index.finish_task(&a, 2).unwrap();
    index
        .track_resource(&b, ResourceKind::Claim, "c".into())
        .unwrap();
    assert_eq!(
        index.start_session(a.clone(), None),
        Err(IndexError::RetiredIncarnation)
    );
    let mut other = b.clone();
    other.incarnation = "new".into();
    assert_eq!(
        index.start_session(other, None),
        Err(IndexError::ActiveIncarnation)
    );
    assert_eq!(index.finish_task(&b, 1).unwrap().claims.len(), 1);
    assert_eq!(index.usage().claims, 0);
    let mut huge = session("x");
    huge.session_id = "x".repeat(129);
    assert_eq!(
        index.start_session(huge, None),
        Err(IndexError::InvalidIdentity)
    );
}

#[test]
fn oversized_explicit_prefix_is_not_executable() {
    let mut index = SessionIndex::new(limits());
    let large = register(&mut index, "too-large-for-p1", None, 65_540, None);
    let target = index.resolve(Selector::Prefix {
        content: large,
        start_token: 0,
        end_token: 65_540,
    });
    assert!(target.materialization_required().iter().any(|gap| matches!(
        gap,
        MaterializationGap::ProtocolTokenLimit {
            end_token: 65_540,
            max_tokens: 65_536
        }
    )));
    assert!(index.materialize(target.known_scope()[0].scope()).is_err());
}

#[test]
fn retirement_reserves_cleanup_and_admits_unpinned_sessions_across_fifo_rollovers() {
    let mut config = limits();
    const LIMIT: usize = 1024;
    config.max_tombstones = LIMIT;
    config.max_sessions = LIMIT + 1;
    let mut index = SessionIndex::new(config);
    let pinned = SessionIdentity {
        retirement_generation: Some(0),
        ..session("pinned")
    };
    let live = std::iter::once(pinned.clone())
        .chain((1..LIMIT).map(|n| session(&format!("live-{n}"))))
        .collect::<Vec<_>>();
    for id in &live {
        index.start_session(id.clone(), None).unwrap();
    }
    assert_eq!(
        index.start_session(session("overflow"), None),
        Err(IndexError::Capacity("tombstones"))
    );
    for id in &live {
        index.finish_task(id, 1).unwrap();
    }
    for n in 1..=10 * LIMIT {
        let id = session(&format!("task-{n}"));
        index
            .start_session(id.clone(), None)
            .unwrap_or_else(|err| panic!("task-{n} admission failed: {err:?}"));
        let claim = index
            .track_resource(&id, ResourceKind::Claim, "priority-only-pin".into())
            .unwrap();
        assert_eq!(
            index.finish_task(&id, 0).unwrap_err(),
            IndexError::StaleRevision
        );
        assert_eq!(index.finish_task(&id, 1).unwrap().claims, [claim]);
        assert_eq!(index.usage().claims, 0);
        assert_eq!(index.usage().tombstones, LIMIT);
        assert_eq!(
            index.start_session(id, None),
            Err(IndexError::RetiredIncarnation)
        );
    }

    let generation = index.retirement_generation();
    assert!(generation > 1);
    assert_eq!(
        index.start_session(pinned, None),
        Err(IndexError::RetiredIncarnation)
    );
    let mut new = session("new-pinned");
    for invalid in [generation - 1, generation + 1] {
        new.retirement_generation = Some(invalid);
        assert_eq!(
            index.start_session(new.clone(), None),
            Err(IndexError::RetiredIncarnation)
        );
    }
    new.retirement_generation = Some(generation);
    index.start_session(new, None).unwrap();
}

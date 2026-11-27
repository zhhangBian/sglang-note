//! Validation for the Engine public action wire contract.
use crate::models::PrefixTarget;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;

pub const PROTOCOL_VERSION: &str = "0.1";
pub const ACTION_VERSION: &str = "1.0";
pub const HASH_SCHEME: &str = "sglang.sha256.u32le.i64.v1";
pub const MAX_INTEGER: u64 = (1 << 53) - 1;
pub(crate) const MAX_SCOPE_TOKENS: usize = 65_536;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Envelope {
    pub protocol_version: String,
    pub message_id: String,
    #[serde(default)]
    pub actions: Vec<Action>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Action {
    pub action_id: String,
    pub action_type: String,
    pub action_version: String,
    #[serde(default = "empty_payload")]
    pub payload: Value,
}
fn empty_payload() -> Value {
    serde_json::json!({})
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Location {
    pub tier: String,
    pub engine_id: Option<String>,
    pub backend: Option<String>,
    pub backend_namespace: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Transfer {
    // The Engine accepts concrete prefixes; the Router resolves session selectors.
    pub target: Value,
    pub source: Location,
    pub destination: Location,
    #[serde(default)]
    pub allow_partial: bool,
    pub timeout_ms: Option<u64>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pin {
    pub mode: String,
    pub target: Value,
    pub location: Location,
    pub claim_id: String,
    pub revision: u64,
    pub ttl_seconds: Option<f64>,
    pub priority: Option<i32>,
    pub required_protection: Option<String>,
}
#[derive(Clone, Debug, PartialEq)]
pub enum ParsedAction {
    Prefetch(Transfer),
    Demote(Transfer),
    Pin(Pin),
    /// Independent consumer payload is not inspected or rewritten.
    Unhandled(Action),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidPayload,
    Unsupported,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TokensTarget {
    kind: String,
    namespace: String,
    token_ids: Vec<u64>,
    #[serde(default)]
    start_token: u64,
    end_token: Option<u64>,
    cache_salt: Option<String>,
    extra_key: Option<String>,
    layout_id: Option<String>,
    #[serde(default, rename = "is_multimodal")]
    _is_multimodal: bool,
    #[serde(default, rename = "allow_clip")]
    _allow_clip: bool,
}

pub fn parse_action(action: &Action) -> Result<ParsedAction, Error> {
    if !action.action_type.starts_with("sglang.") {
        return Ok(ParsedAction::Unhandled(action.clone()));
    }
    if action.action_version != ACTION_VERSION {
        return Err(Error::Unsupported);
    }
    parse_engine_action(&action.action_type, &action.payload)
}

pub(crate) fn parse_engine_action(kind: &str, payload: &Value) -> Result<ParsedAction, Error> {
    use Error::{InvalidPayload, Unsupported};
    if !matches!(kind, "sglang.pin" | "sglang.prefetch" | "sglang.demote") {
        return Err(Unsupported);
    }
    let decode_error = |_| InvalidPayload;
    if kind == "sglang.pin" {
        let pin = Pin::deserialize(payload).map_err(decode_error)?;
        validate_target(&pin.target)?;
        validate_location(&pin.location)?;
        if !matches!(pin.mode.as_str(), "acquire" | "renew" | "release")
            || !valid_text(&pin.claim_id)
            || pin.revision > MAX_INTEGER
            || pin
                .ttl_seconds
                .is_some_and(|ttl| ttl <= 0.0 || ttl > 31_536_000.0)
            || pin
                .required_protection
                .as_deref()
                .is_some_and(|p| !matches!(p, "eviction_exclusion" | "eviction_priority"))
        {
            return Err(InvalidPayload);
        }
        if pin.location.tier != "host" {
            return Err(Unsupported);
        }
        if (pin.mode == "release") != (pin.ttl_seconds.is_none() && pin.priority.is_none()) {
            return Err(InvalidPayload);
        }
        return Ok(ParsedAction::Pin(pin));
    }
    let transfer = Transfer::deserialize(payload).map_err(decode_error)?;
    validate_target(&transfer.target)?;
    validate_location(&transfer.source)?;
    validate_location(&transfer.destination)?;
    if transfer
        .timeout_ms
        .is_some_and(|ms| !(1..=86_400_000).contains(&ms))
    {
        return Err(InvalidPayload);
    }
    match (
        kind,
        transfer.source.tier.as_str(),
        transfer.destination.tier.as_str(),
    ) {
        ("sglang.prefetch", "storage", "host") => Ok(ParsedAction::Prefetch(transfer)),
        ("sglang.demote", "host", "storage") => Ok(ParsedAction::Demote(transfer)),
        (_, "gpu", _) | (_, _, "gpu") => Err(Unsupported),
        _ => Err(InvalidPayload),
    }
}

fn valid_text(text: &str) -> bool {
    (1..=256).contains(&text.chars().count())
}

fn validate_location(location: &Location) -> Result<(), Error> {
    let storage = location.tier == "storage";
    if !matches!(location.tier.as_str(), "gpu" | "host" | "storage")
        || (storage && location.engine_id.is_some())
        || (!storage && (location.backend.is_some() || location.backend_namespace.is_some()))
        || [
            &location.engine_id,
            &location.backend,
            &location.backend_namespace,
        ]
        .into_iter()
        .flatten()
        .any(|s| !valid_text(s))
    {
        return Err(Error::InvalidPayload);
    }
    Ok(())
}

fn validate_target(target: &Value) -> Result<(), Error> {
    if target["kind"] == "prefix" {
        return validate_prefix(target).map(drop);
    }
    let tokens = TokensTarget::deserialize(target).map_err(|_| Error::InvalidPayload)?;
    if tokens.kind != "tokens"
        || !valid_text(&tokens.namespace)
        || !(1..=MAX_SCOPE_TOKENS).contains(&tokens.token_ids.len())
        || tokens.token_ids.iter().any(|&id| id > MAX_INTEGER)
        || tokens.start_token > MAX_INTEGER
        || tokens
            .end_token
            .is_some_and(|end| end > MAX_INTEGER || end <= tokens.start_token)
        || [tokens.cache_salt, tokens.extra_key, tokens.layout_id]
            .into_iter()
            .flatten()
            .any(|s| !valid_text(&s))
    {
        return Err(Error::InvalidPayload);
    }
    Ok(())
}

pub(crate) fn validate_prefix(target: &Value) -> Result<PrefixTarget, Error> {
    let prefix = PrefixTarget::deserialize(target).map_err(|_| Error::InvalidPayload)?;
    let c = &prefix.compatibility;
    let page = c.page_size;
    if prefix.kind != "prefix"
        || ![&c.namespace, &c.layout_id, &c.hash_scheme]
            .into_iter()
            .all(|s| valid_text(s))
        || target["compatibility"].get("cache_salt").is_none()
        || c.cache_salt
            .as_ref()
            .is_some_and(|s| s.chars().count() > 256)
        || !(1..=256).contains(&page)
        || !(1..=64).contains(&c.components.len())
        || c.components.iter().any(|s| !valid_text(s))
        || c.components.iter().collect::<HashSet<_>>().len() != c.components.len()
        || prefix.start_token >= prefix.end_token
        || prefix.end_token > MAX_INTEGER
        || prefix.start_token % page != 0
        || prefix.end_token % page != 0
        || prefix
            .token_ids
            .as_ref()
            .is_some_and(|ids| ids.len() > MAX_SCOPE_TOKENS || ids.len() as u64 != prefix.end_token)
    {
        return Err(Error::InvalidPayload);
    }
    Ok(prefix)
}

/// Raw salt in; encoded salt out. Empty and null deliberately share Python semantics.
/// This helper has no auto-detection: callers must apply it exactly once.
pub fn encode_cache_salt(namespace: &str, salt: Option<&str>) -> String {
    let raw = salt.filter(|s| !s.is_empty());
    let bytes =
        serde_json::to_vec(&("sglang.kv.weight.v1", namespace, raw)).expect("UTF-8 strings");
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ClaimState, ClaimStatus, PublicResult};
    use serde_json::{from_value, json};

    #[test]
    fn action_contract() {
        use Error::{InvalidPayload as Invalid, Unsupported};
        let prefix = json!({"kind":"prefix","start_token":0,"end_token":4,"token_ids":[1,2,3,4],
            "compatibility":{"namespace":"ns","cache_salt":null,"layout_id":"full:p4",
                "page_size":4,"components":["full"],"hash_scheme":HASH_SCHEME}});
        let tokens = json!({"kind":"tokens","namespace":"ns","token_ids":[MAX_INTEGER]});
        let transfer =
            json!({"target":prefix,"source":{"tier":"storage"},"destination":{"tier":"host"}});
        let demote =
            json!({"target":tokens,"source":{"tier":"host"},"destination":{"tier":"storage"}});
        let pin = json!({"target":prefix,"location":{"tier":"host"},"claim_id":"keep","mode":"acquire","revision":0,"ttl_seconds":0.5});
        #[rustfmt::skip]
        let cases = [
            ("sglang.prefetch", &transfer, "", Value::Null, Ok(())),
            ("sglang.demote", &demote, "", Value::Null, Ok(())),
            ("sglang.pin", &pin, "", Value::Null, Ok(())),
            ("sglang.prefetch", &transfer, "/target", tokens, Ok(())),
            ("sglang.prefetch", &transfer, "/target", json!("request_cached"), Err(Invalid)),
            ("sglang.prefetch", &transfer, "/target/compatibility/page_size", json!(0), Err(Invalid)),
            ("sglang.prefetch", &transfer, "/target/compatibility/components", json!(["full","full"]), Err(Invalid)),
            ("sglang.prefetch", &transfer, "/target/end_token", json!(MAX_INTEGER + 1), Err(Invalid)),
            ("sglang.prefetch", &transfer, "/target/token_ids", json!([1,2,3,u64::from(u32::MAX)+1]), Err(Invalid)),
            ("sglang.prefetch", &transfer, "/target/blocks", json!([]), Err(Invalid)),
            ("sglang.prefetch", &transfer, "/timeout_ms", json!(1.0), Err(Invalid)),
            ("sglang.prefetch", &transfer, "/timeout_ms", json!(86_400_001), Err(Invalid)),
            ("sglang.prefetch", &transfer, "/destination", json!({}), Err(Invalid)),
            ("sglang.prefetch", &transfer, "/destination/tier", json!("gpu"), Err(Unsupported)),
            ("sglang.prefetch", &transfer, "/destination/tier", json!("disk"), Err(Invalid)),
            ("sglang.prefetch", &transfer, "/source", json!({"tier":"storage","engine_id":"e"}), Err(Invalid)),
            ("sglang.pin", &pin, "/ttl_seconds", json!(0), Err(Invalid)),
            ("sglang.pin", &pin, "/priority", json!(i64::from(i32::MAX)+1), Err(Invalid)),
            ("sglang.pin", &pin, "/required_protection", json!("unknown"), Err(Invalid)),
            ("sglang.pin", &pin, "/mode", json!("release"), Err(Invalid)),
            ("sglang.pin", &pin, "/owner_task", Value::Null, Err(Invalid)),
            ("sglang.future", &pin, "", Value::Null, Err(Unsupported)),
        ];
        for (kind, base, path, replacement, expected) in cases {
            let mut payload = base.clone();
            if let Some((parent, key)) = path.rsplit_once('/') {
                payload.pointer_mut(parent).unwrap()[key] = replacement;
            }
            let action = Action {
                action_id: "a".into(),
                action_type: kind.into(),
                action_version: ACTION_VERSION.into(),
                payload,
            };
            assert_eq!(parse_action(&action).map(drop), expected, "{kind} {path}");
        }
        let action = Action {
            action_id: "a".into(),
            action_type: "sglang.pin".into(),
            action_version: "9".into(),
            payload: Value::Null,
        };
        assert_eq!(parse_action(&action), Err(Unsupported));
        let other = Action {
            action_type: "other.future".into(),
            ..action
        };
        assert_eq!(
            parse_action(&other),
            Ok(ParsedAction::Unhandled(other.clone()))
        );
    }

    #[test]
    fn results_and_claims_keep_separate_wire_shapes() {
        let scope = json!([{"coverage":{"start_token":0,"end_token":4,"component":"full"},"status":"partial"}]);
        let result: PublicResult = from_value(json!({"owner_epoch":"epoch","state_version":2,
            "outcome_sequence":2,"ready_scope":scope,"preserved_scope":scope}))
        .unwrap();
        assert_eq!(result.ready_scope, result.preserved_scope);
        assert!(result.claim.is_none());
        let mut claim = json!({"caller_id":"router","task_incarnation":"task-1","claim_id":"keep",
            "revision":3,"state":"released","owner_epoch":"epoch","tombstone":true});
        let parsed: ClaimState = from_value(claim.clone()).unwrap();
        assert!(parsed.tombstone);
        assert_eq!(parsed.state, ClaimStatus::Released);
        assert_eq!(
            serde_json::to_value(parsed).unwrap()["protection_version"],
            "1.0"
        );
        assert!(from_value::<PublicResult>(claim.clone()).is_err());
        claim["state"] = json!("completed");
        assert!(from_value::<ClaimState>(claim).is_err());
        assert!(
            from_value::<PublicResult>(json!({"state_version":1,"outcome_sequence":1})).is_err()
        );
    }

    #[test]
    fn salt_matches_engine_hashes() {
        let empty = "f06d7376b149ed3ad521a51027b5f1e9f956206090aa2aa70d278a0101c87eed";
        #[rustfmt::skip]
        let cases = [
            ("ns", None, empty),
            ("ns", Some(""), empty),
            ("ns", Some("tenant"),
                "dddf3db6cf3c0a983092769f0bc478164da6c2b01469cc218da5fd524bedb2f2"),
            ("权重😀", Some("租户\0\"\\\n"),
                "6cce2ac5ec408ea8460c4a7c6eff285b4f7126e0ba2024ac8f4047b5e52bf58b"),
            ("ns", Some("é"),
                "332501b2526ba1f80fadaf43a4d8d6da1a9160b5a9c05062b6ef7b246b2bfaf5"),
            ("ns", Some("e\u{301}"),
                "07b718dbdd7012bc7e0073124c5254159f774db56c02452b8bffd851ab5f07c1"),
        ];
        for (namespace, salt, hash) in cases {
            assert_eq!(encode_cache_salt(namespace, salt), hash);
        }
    }
}

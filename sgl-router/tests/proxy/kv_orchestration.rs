//! Real Router + real HTTP Engine mocks; no live GPU or external services.
use crate::common::cache_aware_fixture::{config, MODEL};
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use sgl_router::{
    config::{AffinityConfig, CachePrefixProvider},
    discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec},
    orchestration::OrchestrationConfig,
    policies::factory::build_registry,
    policies_reorg::factory::build_resolver,
    proxy::Proxy,
    server::{
        app::build_router,
        app_context::{AppContext, ChatRouting},
    },
    state::kv_events::{
        compute_block_hashes, HashTree, KvEventIndex, KvWorkerId, RadixTreePrefixProvider,
    },
    tokenizer::TokenizerRegistry,
    workers::WorkerRegistry,
};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex, MutexGuard},
    time::Duration,
};
use tokio::task::JoinHandle;

#[derive(Default)]
struct EngineState {
    operations: BTreeMap<String, Value>,
    commands: Vec<Value>,
    inference: Vec<Value>,
    resident: u64,
    running: bool,
    cancelled: bool,
    claims: BTreeMap<String, Value>,
    pin_missing: bool,
    page_size: Option<u64>,
    epoch: String,
    fail_capabilities: bool,
    validate_context: bool,
    cache_epoch: String,
    cache_version: u64,
    omit_discovery_watermark: bool,
    include_claim: bool,
    fail_claim: bool,
    native_generation: u64,
    residency_override: Option<Value>,
    delayed_residency: Option<Arc<Delay>>,
    delayed_capabilities: Option<Arc<Delay>>,
    delayed_submit: Option<Arc<Delay>>,
}
#[derive(Default)]
struct Delay {
    started: tokio::sync::Notify,
    release: tokio::sync::Notify,
}
impl Delay {
    async fn pause(&self) {
        self.started.notify_one();
        self.release.notified().await;
    }

    async fn entered(&self) {
        within(self.started.notified()).await;
    }

    fn unblock(&self) {
        self.release.notify_one();
    }
}

fn cache_watermark(state: &EngineState) -> Value {
    json!({"source_id":"cache:mock","source_epoch":state.cache_epoch,"generation":state.native_generation,"version":state.cache_version})
}
struct Engine {
    url: String,
    state: Arc<Mutex<EngineState>>,
    server: JoinHandle<()>,
}
impl Drop for Engine {
    fn drop(&mut self) {
        self.server.abort();
    }
}
type SharedEngine = Arc<Mutex<EngineState>>;
type EngineReply = (StatusCode, Json<Value>);

fn reply(data: Value) -> EngineReply {
    (StatusCode::OK, Json(json!({"success":true,"data":data})))
}

fn error(status: StatusCode, code: &str) -> EngineReply {
    (status, Json(json!({"success":false,"error":{"code":code}})))
}

async fn capabilities(State(state): State<SharedEngine>) -> EngineReply {
    let delayed = state.lock().unwrap().delayed_capabilities.take();
    if let Some(delayed) = delayed {
        delayed.pause().await;
    }
    let s = state.lock().unwrap();
    if s.fail_capabilities {
        return error(StatusCode::SERVICE_UNAVAILABLE, "unavailable");
    }
    let mut data = json!({"engine_id":"mock-engine","engine_epoch":s.epoch,"namespace":"weights-v1",
        "cache_salt_encoding":"owner_assertion","executor":{"cache_generation":1},"public_protocol":{"actions":{"sglang.prefetch":["1.0"],"sglang.demote":["1.0"],"sglang.pin":["1.0"]},
        "paths":{"sglang.prefetch":[["storage","host"],["host","gpu"]],"sglang.demote":[["host","storage"]]},
        "pin_protections":[{"tier":"host","condition":"eviction_exclusion"},{"tier":"host","condition":"eviction_priority"}],
        "storage_backend":"mooncake","backend_namespace":"pool-a","prefix":{"namespace":"weights-v1","layout_id":"full:p1","page_size":s.page_size.unwrap_or(1),"components":["full"],"hash_scheme":"sglang.sha256.u32le.i64.v1"}}});
    if !s.omit_discovery_watermark {
        data["cache_watermark"] = cache_watermark(&s);
    }
    reply(data)
}
async fn command(State(state): State<SharedEngine>, Json(body): Json<Value>) -> EngineReply {
    let (delay, response) = {
        let mut state = state.lock().unwrap();
        let delay = match body["kind"].as_str() {
            Some("residency") => state.delayed_residency.take(),
            Some("submit") => state.delayed_submit.take(),
            _ => None,
        };
        (delay, command_now(&mut state, body))
    };
    if let Some(delay) = delay {
        delay.pause().await;
    }
    response
}

fn command_now(s: &mut EngineState, body: Value) -> EngineReply {
    let context = if body["kind"] == "cancel" {
        &body["ref"]["context"]
    } else {
        &body["context"]
    };
    if s.validate_context && context["engine_epoch"] != s.epoch {
        return error(StatusCode::CONFLICT, "stale_epoch");
    }
    s.commands.push(body.clone());
    assert!(matches!(
        body["kind"].as_str(),
        Some("submit" | "query" | "claim" | "cancel" | "residency")
    ));
    if body["kind"] == "claim" {
        if s.fail_claim {
            return error(StatusCode::SERVICE_UNAVAILABLE, "unavailable");
        }
        return match s.claims.get(body["claim_id"].as_str().unwrap()) {
            Some(claim) => {
                let mut claim = claim.clone();
                claim["cache_watermark"] = cache_watermark(s);
                reply(claim)
            }
            None => error(StatusCode::NOT_FOUND, "missing"),
        };
    }
    if body["kind"] == "residency" {
        assert_eq!(body["target"]["kind"], "tokens");
        let end = s.resident.min(token_count(&body["target"]));
        let items: Vec<_> = (end > 0)
            .then(|| {
                json!({"coverage_kind":"tokens",
            "descriptor":{"component":"full","start_token":0,"end_token":end,"dependency_start":0},
            "location":body["location"]})
            })
            .into_iter()
            .collect();
        let mark = s
            .residency_override
            .clone()
            .unwrap_or_else(|| cache_watermark(s));
        return reply(
            json!({"namespace":"weights-v1","engine_epoch":s.epoch,"cache_generation":1,
            "observed_at":1,"items":items,"missing":[],"cache_watermark":mark}),
        );
    }
    if body["kind"] == "submit" {
        let id = body["envelope"]["message_id"].as_str().unwrap().to_string();
        let action = action(&body);
        let target = &action["payload"]["target"];
        assert_eq!(target["kind"], "prefix");
        assert_eq!(action["action_version"], "1.0");
        let count = token_count(target);
        let pin = action["action_type"] == "sglang.pin";
        let status = if s.running {
            "running"
        } else if pin && s.pin_missing {
            "missing"
        } else {
            "completed"
        };
        let mut result = json!({"ref":{"context":body["context"],"message_id":id,"action_id":"transfer","operation_id":format!("op-{id}")},
            "sequence":1,"update":{"status":status,"quiescent":!s.running},"public_result":{"state_version":99_999}});
        if pin && !s.pin_missing {
            let payload = &action["payload"];
            if status == "completed" && payload["mode"] == "release" {
                result["update"]["completed_steps"] = json!(["protection_released"]);
            }
            let epoch = s.epoch.clone();
            s.claims.insert(payload["claim_id"].as_str().unwrap().into(),json!({"owner_epoch":epoch,"revision":payload["revision"],
                "active":payload["mode"]!="release","ttl_remaining_ms":null,"state":if payload["mode"]=="release"{"released"}else{"active"},"priority":payload["priority"],
                "effective_protection":if payload["ttl_seconds"].is_null(){"eviction_priority"}else{"eviction_exclusion"},
                "actual_scope":[{"start_token":0,"end_token":count}]}));
        }
        s.operations.entry(id).or_insert(result);
        if !s.running && action["action_type"] == "sglang.prefetch" {
            s.resident = count;
        }
    }
    if body["kind"] == "cancel" {
        s.cancelled = true;
        let id = body["ref"]["message_id"].as_str().unwrap();
        let result = s.operations.get_mut(id).unwrap();
        result["sequence"] = json!(2);
        result["update"] = json!({"status":"cancelled","quiescent":true});
    }
    let id = body["message_id"]
        .as_str()
        .or_else(|| body["envelope"]["message_id"].as_str())
        .or_else(|| body["ref"]["message_id"].as_str())
        .unwrap_or("");
    match s.operations.get(id) {
        Some(r) => {
            let mut data = json!({"results":[r],"cache_watermark":cache_watermark(s)});
            if let Some(id) = body["include_claim_id"]
                .as_str()
                .filter(|_| s.include_claim)
            {
                data["claim"] = json!(s.claims.get(id));
            }
            reply(data)
        }
        None => error(StatusCode::NOT_FOUND, "missing"),
    }
}
fn token_count(target: &Value) -> u64 {
    target["token_ids"].as_array().unwrap().len() as u64
}

async fn inference(
    State(state): State<SharedEngine>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Json<Value> {
    assert!(!headers.contains_key("x-kv-control-authorization"));
    state.lock().unwrap().inference.push(body);
    Json(
        json!({"id":"mock","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}),
    )
}
impl Engine {
    fn state(&self) -> MutexGuard<'_, EngineState> {
        self.state.lock().unwrap()
    }

    fn has_active_claim(&self) -> bool {
        self.state().claims.values().any(|c| c["active"] == true)
    }

    async fn start() -> Self {
        let state = Arc::new(Mutex::new(EngineState {
            epoch: "epoch-1".into(),
            cache_version: 1,
            cache_epoch: "cache-1".into(),
            native_generation: 1,
            ..Default::default()
        }));
        let app = Router::new()
            .route("/v1/kv/capabilities", get(capabilities))
            .route("/v1/kv/control", post(command))
            .route("/generate", post(inference))
            .route("/v1/chat/completions", post(inference))
            .with_state(state.clone());
        let (url, server) = serve(app).await;
        Self { url, state, server }
    }
}
struct TestRouter {
    tree: Arc<HashTree>,
    url: String,
    server: JoinHandle<()>,
    client: reqwest::Client,
}
impl Drop for TestRouter {
    fn drop(&mut self) {
        self.server.abort();
    }
}
fn orchestration() -> OrchestrationConfig {
    serde_json::from_value(json!({"caller_id":"trusted-router","control_bearer":"test-control",
    "queue_capacity":32,"max_operations":64,"max_workers":4,"max_engine_bytes":1048576,"rpc_timeout_ms":1000,"poll_ms":50,"operation_ttl_ms":10000,
    "limits":{"max_sessions":16,"max_branches":32,"max_history_revisions":64,"max_requests":64,"max_content_nodes":1024,"max_token_material_bytes":1048576,
        "max_tombstones":32,"max_claims":32,"max_intents":64,"max_identifier_bytes":128,"max_components":8}})).unwrap()
}
async fn router(engine: &Engine) -> TestRouter {
    router_with(&[engine], false, orchestration()).await
}

async fn router_with(
    engines: &[&Engine],
    reorg: bool,
    orchestration: OrchestrationConfig,
) -> TestRouter {
    let mut cfg = config();
    cfg.server.orchestration = Some(orchestration);
    let model = ModelId(cfg.model.id.clone());
    cfg.model.cache_aware.as_mut().unwrap().prefix_provider = CachePrefixProvider::RadixTree;
    cfg.model.affinity = Some(AffinityConfig {
        cache_affinity_min_matched_tokens: Some(0),
        cache_candidate_min_workers: 1,
        cache_candidate_ratio: 1.0,
        cache_candidate_max_workers: 1,
        ..Default::default()
    });
    let registry = Arc::new(WorkerRegistry::default());
    for (i, e) in engines.iter().enumerate() {
        registry
            .add(WorkerSpec {
                id: WorkerId(format!("engine-{i}")),
                url: e.url.clone(),
                model_ids: vec![model.clone()],
                mode: WorkerMode::Plain,
                ..Default::default()
            })
            .unwrap();
    }
    let state = KvEventIndex::new();
    let tree = state.tree();
    let oracle = state.block_size_oracle();
    oracle.try_set(1).unwrap();
    let policies = build_registry(&cfg, tree.clone(), oracle.clone()).unwrap();
    let mut ctx = AppContext::new(
        cfg.clone(),
        Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap()),
        Arc::new(Proxy::new(Duration::from_secs(3)).unwrap()),
        registry,
        Arc::new(policies),
    );
    if reorg {
        let (resolver, _) = build_resolver(&cfg.model, &state, None).unwrap();
        ctx.chat_routing = ChatRouting::Reorg([(model, resolver)].into());
    }
    ctx.radix_tree_prefix_provider = Some(RadixTreePrefixProvider::new(tree.clone(), oracle));
    let (url, server) = serve(build_router(Arc::new(ctx))).await;
    TestRouter {
        tree,
        url,
        server,
        client: reqwest::Client::new(),
    }
}

async fn serve(app: Router) -> (String, JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (url, server)
}
const READ: &str = "prepare";
const PIN: &str = "keep";
const TOKENS: [u32; 4] = [1, 2, 3, 4];

async fn within<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(3), future)
        .await
        .expect("request or poll timed out")
}

async fn poll<T>(mut check: impl AsyncFnMut() -> Option<T>) -> T {
    within(async {
        loop {
            if let Some(value) = check().await {
                return value;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
}

impl TestRouter {
    fn post(&self, path: &str, body: Value) -> reqwest::RequestBuilder {
        self.client.post(format!("{}{path}", self.url)).json(&body)
    }

    fn control(&self, body: Value) -> reqwest::RequestBuilder {
        self.post("/v1/kv/control", body)
            .bearer_auth("test-control")
    }

    async fn api(&self, body: Value) -> Value {
        control_data(self.control(body)).await
    }

    async fn ack(&self, task: &Value, message: &str) {
        let reply = within(self.operation("ack", task, message)).await;
        assert_eq!(reply["acknowledged"], true);
    }

    fn generate(&self, task: &Value, revision: u64, ids: &[u32]) -> reqwest::RequestBuilder {
        self.post(
            "/generate",
            json!({"input_ids":ids,"kv_context":context(task, revision)}),
        )
    }

    async fn drained(&self, task: &Value, message: &str) {
        poll(async || {
            (self.operation("query", task, message).await["quiescent"] == true).then_some(())
        })
        .await;
    }

    async fn operation(&self, kind: &str, task: &Value, message: &str) -> Value {
        self.api(operation(kind, task, message)).await
    }

    async fn event(&self, task: &Value, event: &str, revision: u64) -> Value {
        self.api(lifecycle(task, event, revision)).await
    }

    async fn ready(&self, task: &Value, message: &str) -> Value {
        poll(async || {
            let view = self.operation("query", task, message).await;
            op(&view)["ready_scope"].is_object().then_some(view)
        })
        .await
    }
}

fn task(id: &str) -> Value {
    json!({"session_id":id,"incarnation":"task-1"})
}

fn context(task: &Value, revision: u64) -> Value {
    let mut context = json!({"branch_id":"main","revision":revision,"branch_revision":revision,
        "request_id":"r","request_incarnation":format!("request-{revision}")});
    context
        .as_object_mut()
        .unwrap()
        .extend(task.as_object().unwrap().clone());
    context
}

fn lifecycle(task: &Value, event: &str, revision: u64) -> Value {
    json!({"kind":"session_event","event":event,"task":task,"revision":revision,"event_id":event})
}

fn operation(kind: &str, task: &Value, message: &str) -> Value {
    json!({"kind":kind,"task":task,"message_id":message})
}

fn prefix(tokens: &[u32]) -> Value {
    json!({"kind":"prefix","compatibility":{"namespace":"weights-v1","cache_salt":null,"layout_id":"full:p1",
        "page_size":1,"components":["full"],"hash_scheme":"sglang.sha256.u32le.i64.v1"},
        "start_token":0,"end_token":tokens.len(),"token_ids":tokens})
}

fn prefetch(task: &Value, message: &str, target: Value, engine: &Engine) -> Value {
    json!({"kind":"submit","task":task,"message_id":message,"action":"sglang.prefetch","target":target,
        "worker_url":engine.url,"source":{"tier":"storage"},"destination":{"tier":"host"}})
}

fn pin(task: &Value, target: Value, priority: i32) -> Value {
    json!({"kind":"submit","task":task,"message_id":PIN,"action":"sglang.pin","target":target,
        "location":{"tier":"host"},"mode":"acquire","claim_id":PIN,"revision":1,"priority":priority})
}

fn action(command: &Value) -> &Value {
    &command["envelope"]["actions"][0]
}

fn op(view: &Value) -> &Value {
    &view["operations"][0]
}

fn gated_config() -> OrchestrationConfig {
    OrchestrationConfig {
        rpc_timeout_ms: 8_000,
        ..orchestration()
    }
}

fn scoped_config() -> OrchestrationConfig {
    let mut cfg = orchestration();
    cfg.poll_ms = cfg.operation_ttl_ms;
    cfg
}

async fn json_reply(request: reqwest::RequestBuilder) -> Value {
    within(request.send()).await.unwrap().json().await.unwrap()
}

async fn control_data(request: reqwest::RequestBuilder) -> Value {
    let mut reply = json_reply(request).await;
    assert_eq!(reply["success"], true, "{reply}");
    reply["data"].take()
}

async fn inference_ok(request: reqwest::RequestBuilder) -> reqwest::Response {
    within(request.send())
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
}

#[tokio::test]
async fn lifecycle_uses_passive_association_for_tool_wait_and_prepare() {
    let engine = Engine::start().await;
    let router = router(&engine).await;
    let task = task("chat");
    router.event(&task, "task_started", 0).await;
    inference_ok(router.generate(&task, 1, &TOKENS)).await;
    {
        let state = engine.state();
        assert_eq!(state.inference[0]["input_ids"], json!(TOKENS));
        assert!(state.inference[0].get("kv_context").is_none());
        assert!(
            state.commands.is_empty(),
            "passive metadata cannot trigger control"
        );
    }
    let legacy = prefetch(&task, "legacy", json!("request_cached"), &engine);
    assert_eq!(router.control(legacy).send().await.unwrap().status(), 400);
    for (event, revision, action_type, source, destination) in [
        ("tool_wait", 2, "sglang.demote", "host", "storage"),
        ("prepare", 3, "sglang.prefetch", "storage", "host"),
    ] {
        let mut body = lifecycle(&task, event, revision);
        body["pressure"] = json!(true);
        body["worker_url"] = json!(engine.url);
        control_data(
            router
                .post("/v1/kv/session/events", body)
                .bearer_auth("test-control"),
        )
        .await;
        let receipt = router.operation("query", &task, event).await;
        assert_eq!(op(&receipt)["result"]["update"]["status"], "completed");
        let state = engine.state();
        let dispatched = state
            .commands
            .iter()
            .map(action)
            .find(|a| a["action_type"] == action_type)
            .unwrap();
        assert_eq!(dispatched["payload"]["source"]["tier"], source);
        assert_eq!(dispatched["payload"]["destination"]["tier"], destination);
    }
    assert_eq!(
        op(&router.ready(&task, READ).await)["ready_scope"]["end_token"],
        4
    );
    inference_ok(router.generate(&task, 4, &TOKENS)).await;
    assert_eq!(engine.state().inference.len(), 2);
    assert_eq!(
        router.event(&task, "task_finished", 5).await["state"],
        "cleanup_requested"
    );
    router.ack(&task, READ).await;
}

#[tokio::test]
async fn responses_with_orchestration_returns_output_and_strips_kv_context() {
    let engine = Engine::start().await;
    let router = router(&engine).await;
    let task = task("response");
    let body = json!({"model":MODEL,"input":"reply OK","kv_context":context(&task, 1)});
    let response = inference_ok(router.post("/v1/responses", body)).await;
    let response: Value = response.json().await.unwrap();
    assert_eq!(response["object"], "response");
    assert_eq!(response["status"], "completed");
    assert_eq!(response["output"][0]["content"][0]["text"], "ok");
    let state = engine.state();
    assert_eq!(state.inference[0]["messages"][0]["content"], "reply OK");
    assert!(state.inference[0].get("kv_context").is_none());
}

#[tokio::test]
async fn long_input_is_forwarded_in_full_while_prefetch_dispatch_is_clipped_and_async() {
    let engine = Engine::start().await;
    let delay = Arc::new(Delay::default());
    {
        let mut state = engine.state();
        state.page_size = Some(96);
        state.delayed_submit = Some(delay.clone());
    }
    let router = router_with(&[&engine], false, gated_config()).await;
    let task = task("long");
    let ids: Vec<u32> = (0..70_000).collect();
    inference_ok(
        router
            .generate(&task, 1, &ids)
            .header("x-kv-control-authorization", "Bearer test-control"),
    )
    .await;
    delay.entered().await;
    const END: usize = 65_472;
    {
        let state = engine.state();
        assert_eq!(state.inference[0]["input_ids"], json!(ids));
        assert!(state.inference[0].get("kv_context").is_none());
        let submitted = state
            .commands
            .iter()
            .map(action)
            .find(|a| a["action_type"] == "sglang.prefetch")
            .unwrap();
        let target = &submitted["payload"]["target"];
        assert_eq!(target["end_token"], END);
        assert_eq!(target["token_ids"], json!(ids[..END]));
    }
    delay.unblock();
    let branch = json!({"kind":"branch","branch_id":"main"});
    let view = router
        .api(prefetch(&task, "inspect", branch, &engine))
        .await;
    assert_eq!(view["known_scope"][0]["controlled_end"], END);
    assert_eq!(view["known_scope"][0]["original_end"], ids.len());
}

#[tokio::test]
async fn pin_cleanup_handles_finish_missing_claim_and_retry_after_failed_lookup() {
    for (missing, retry) in [(false, false), (true, false), (false, true)] {
        let engine = Engine::start().await;
        engine.state().pin_missing = missing;
        let router = router_with(&[&engine], false, scoped_config()).await;
        let task = task("pin");
        router.event(&task, "task_started", 0).await;
        router
            .api(pin(&task, prefix(&TOKENS), i32::from(retry)))
            .await;
        let view = router.operation("query", &task, PIN).await;
        let first = op(&view);
        assert_eq!(
            first["result"]["update"]["status"],
            if missing { "missing" } else { "completed" }
        );
        assert!(first["ready_scope"].is_null());
        if !missing {
            assert_eq!(first["claim"]["state"], "active");
            assert_eq!(first["claim"]["priority"], i32::from(retry));
        }
        if retry {
            engine.state().fail_claim = true;
            router.operation("cancel", &task, PIN).await;
            assert_eq!(
                router.operation("query", &task, PIN).await["quiescent"],
                false
            );
            let mut state = engine.state();
            assert!(state.claims.values().any(|c| c["state"] == "active"));
            state.fail_claim = false;
        } else {
            router.event(&task, "task_finished", 1).await;
        }
        router.drained(&task, PIN).await;

        // Confirmed cleanup can be acknowledged independently of discovery.
        engine.state().fail_capabilities = true;
        let view = router.operation("query", &task, PIN).await;
        assert_eq!(view["quiescent"], true);
        if missing {
            assert!(op(&view)["claim"].is_null());
            assert!(op(&view)["error"].is_null());
        }
        router.ack(&task, PIN).await;
        let state = engine.state();
        assert!(state.claims.values().all(|c| c["active"] == false));
        assert!(state.commands.iter().any(|c| c["kind"] == "claim"));
        let pins: Vec<_> = state
            .commands
            .iter()
            .filter(|c| c["kind"] == "submit")
            .map(|c| &action(c)["payload"])
            .collect();
        assert_eq!(pins.len(), if missing { 1 } else { 2 });
        assert!(pins.iter().all(|p| p.get("owner_task").is_none()));
        if !missing {
            assert_eq!(pins[1]["mode"], "release");
            assert!(pins[1]["priority"].is_null() && pins[1]["ttl_seconds"].is_null());
        }
    }
}

#[tokio::test]
async fn real_pins_release_across_retirement_generations() {
    let engine = Engine::start().await;
    let mut cfg = scoped_config();
    cfg.limits.max_tombstones = 1;
    cfg.limits.max_claims = 1;
    let router = router_with(&[&engine], false, cfg).await;
    let mut first = Value::Null;
    for n in 0..6 {
        let mut task = task(&format!("task-{n}"));
        if n % 2 == 0 {
            task["retirement_generation"] =
                router.api(json!({"kind":"capabilities"})).await["retirement_generation"].clone();
        }
        let started = lifecycle(&task, "task_started", 0);
        if n == 0 {
            first = started.clone();
        }
        if n != 3 {
            router.api(started).await;
        }
        inference_ok(router.post("/v1/chat/completions", json!({"model":MODEL,
            "messages":[{"role":"user","content":"hello"}],"input_ids":TOKENS,"kv_context":context(&task, 1)}))).await;
        let view = router
            .api(pin(&task, json!({"kind":"branch","branch_id":"main"}), 3))
            .await;
        assert_eq!(view["known_scope"].as_array().unwrap().len(), 1);
        poll(async || {
            (op(&router.operation("query", &task, PIN).await)["claim"]["active"] == true)
                .then_some(())
        })
        .await;
        let ended = router.event(&task, "task_finished", 2).await;
        assert_eq!(ended["claims"].as_array().unwrap().len(), 1);
        assert_eq!(
            ended["claims"][0]["task"].get("retirement_generation"),
            task.get("retirement_generation")
        );
        router.drained(&task, PIN).await;
        router.ack(&task, PIN).await;
        assert!(!engine.has_active_claim());
    }
    let replay = json_reply(router.control(first)).await;
    assert_eq!(replay["error"]["reason"], "RetiredIncarnation");
}

#[tokio::test]
async fn failed_discovery_does_not_prove_owner_loss_but_replacement_epoch_does() {
    for restart in [false, true] {
        let engine = Engine::start().await;
        engine.state().running = true;
        let router = router_with(
            &[&engine],
            false,
            OrchestrationConfig {
                max_operations: 1,
                ..scoped_config()
            },
        )
        .await;
        let task = task("owner");
        router.event(&task, "task_started", 0).await;
        router
            .api(prefetch(&task, READ, prefix(&TOKENS), &engine))
            .await;
        let running = router.operation("query", &task, READ).await;
        assert_eq!(op(&running)["result"]["update"]["status"], "running");
        {
            let mut state = engine.state();
            state.fail_capabilities = true;
            state.validate_context = true;
            if restart {
                state.epoch = "epoch-2".into();
            }
        }
        router.operation("cancel", &task, READ).await;
        let view = router.operation("query", &task, READ).await;
        assert!(op(&view)["ready_scope"].is_null());
        assert_ne!(op(&view)["owner_lost"], true);
        if restart {
            assert_eq!(view["quiescent"], false);
            assert!(op(&view)["error"].as_str().unwrap().contains("stale_epoch"));
            assert!(!engine.state().cancelled);
            engine.state().fail_capabilities = false;
            let retired = router.operation("query", &task, READ).await;
            assert_eq!(op(&retired)["owner_lost"], true);
            assert_eq!(retired["quiescent"], true);
            assert_eq!(op(&retired)["result"], op(&running)["result"]);
        } else {
            assert_eq!(op(&view)["result"]["update"]["status"], "cancelled");
            assert_eq!(view["quiescent"], true);
            assert!(engine.state().cancelled);
        }
        router.ack(&task, READ).await;
        if restart {
            let next = router
                .api(prefetch(&task, "new", prefix(&TOKENS), &engine))
                .await;
            assert_eq!(op(&next)["owner_epoch"], "epoch-2");
        }
    }
}

#[tokio::test]
async fn inline_watermarks_invalidate_ready_scope_and_reject_stale_residency() {
    let engine = Engine::start().await;
    engine.state().omit_discovery_watermark = true;
    let router = router_with(&[&engine], false, scoped_config()).await;
    let task = task("watermark");
    router.event(&task, "task_started", 0).await;
    let first = prefetch(&task, "a", prefix(&TOKENS), &engine);
    router.api(first.clone()).await;
    let second = prefetch(&task, "b", prefix(&[5, 6, 7, 8]), &engine);
    router.api(second).await;
    for replace_source in [false, true] {
        let before = router.ready(&task, "a").await;
        let old_mark = op(&before)["cache_watermark"].clone();
        {
            let mut state = engine.state();
            state.native_generation += 1;
            state.cache_version += 1;
            if replace_source {
                state.cache_epoch = "replacement".into();
            }
        }
        let current = router.ready(&task, "b").await;
        let mark = &op(&current)["cache_watermark"];
        assert_ne!(*mark, old_mark);
        let old = router.api(first.clone()).await;
        assert!(op(&old)["ready_scope"].is_null());
        assert_eq!(
            router.api(json!({"kind":"feedback"})).await["owners"][0]["cache_watermark"],
            *mark
        );
        let mut stale_mark = mark.clone();
        stale_mark["version"] = old_mark["version"].clone();
        engine.state().residency_override = Some(stale_mark);
        let stale = router.operation("query", &task, "a").await;
        assert!(op(&stale)["ready_scope"].is_null());
        assert_eq!(op(&stale)["result"], op(&before)["result"]);
        engine.state().residency_override = None;
    }
}

#[tokio::test]
async fn late_ack_during_shared_residency_preserves_the_live_terminal_receipt() {
    let engine = Engine::start().await;
    let router = router_with(&[&engine], false, gated_config()).await;
    let finished = task("finished");
    let live = task("live");
    for task in [&finished, &live] {
        router.event(task, "task_started", 0).await;
        let command = prefetch(task, READ, prefix(&TOKENS), &engine);
        router.api(command).await;
    }
    let receipt = router.ready(&finished, READ).await;
    let shared = router.ready(&live, READ).await;
    assert_eq!(op(&receipt)["child_id"], op(&shared)["child_id"]);
    assert_eq!(op(&receipt)["result"]["update"]["status"], "completed");
    router.event(&finished, "task_finished", 1).await;
    assert_eq!(
        router.operation("query", &finished, READ).await["quiescent"],
        true
    );

    let delay = Arc::new(Delay::default());
    let before = {
        let mut state = engine.state();
        state.delayed_residency = Some(delay.clone());
        state.commands.len()
    };
    let ack = async {
        delay.entered().await;
        router.ack(&finished, READ).await;
        delay.unblock();
    };
    let (view, ()) = tokio::join!(router.operation("query", &live, READ), ack);
    assert_eq!(op(&view)["ready_scope"]["end_token"], 4);
    assert_eq!(op(&view)["result"], op(&receipt)["result"]);
    let state = engine.state();
    assert_eq!(
        state.commands[before..]
            .iter()
            .map(|c| c["kind"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["residency"]
    );
}

#[tokio::test]
async fn pin_query_batches_claim_and_falls_back_only_if_omitted() {
    for batched in [false, true] {
        let engine = Engine::start().await;
        {
            let mut state = engine.state();
            state.running = true;
            state.include_claim = batched;
        }
        let router = router_with(&[&engine], false, scoped_config()).await;
        let task = task("batch");
        router.event(&task, "task_started", 0).await;
        router.api(pin(&task, prefix(&TOKENS), 0)).await;
        router.operation("query", &task, PIN).await;
        let before = engine.state().commands.len();
        assert_eq!(
            op(&router.operation("query", &task, PIN).await)["claim"]["active"],
            true
        );
        let state = engine.state();
        let calls = &state.commands[before..];
        assert_eq!(calls.len(), if batched { 1 } else { 2 });
        assert_eq!(calls[0]["kind"], "query");
        assert!(calls[0]["include_claim_id"].is_string());
        if !batched {
            assert_eq!(calls[1]["kind"], "claim");
        }
    }
}

#[tokio::test]
async fn slow_discovery_and_saturated_residency_do_not_block_healthy_worker() {
    for reorg in [false, true] {
        let slow = Engine::start().await;
        let fast = Engine::start().await;
        let discovery = Arc::new(Delay::default());
        slow.state().delayed_capabilities = Some(discovery.clone());
        let router = router_with(
            &[&slow, &fast],
            reorg,
            OrchestrationConfig {
                queue_capacity: 2,
                ..gated_config()
            },
        )
        .await;
        let slow_task = task("slow");
        router.event(&slow_task, "task_started", 0).await;
        router.tree.insert(
            &KvWorkerId::new(fast.url.clone(), 0),
            None,
            &compute_block_hashes(&TOKENS, 1),
        );
        let healthy = async |id: &str| {
            let task = task(id);
            router.event(&task, "task_started", 0).await;
            let command = prefetch(&task, READ, prefix(&TOKENS), &fast);
            router.api(command).await;
            assert_eq!(op(&router.ready(&task, READ).await)["worker_url"], fast.url);
        };
        let healthy_inference = async || {
            let response =
                inference_ok(router.post("/generate", json!({"input_ids":TOKENS}))).await;
            assert_eq!(response.headers()["x-sgl-worker-url"], fast.url);
        };
        let during_discovery = async {
            discovery.entered().await;
            within(healthy("discovery")).await;
            healthy_inference().await;
            discovery.unblock();
        };
        let (submitted, ()) = tokio::join!(
            router.api(prefetch(&slow_task, READ, prefix(&TOKENS), &slow)),
            during_discovery
        );
        assert_eq!(op(&submitted)["worker_url"], slow.url);

        let residency = Arc::new(Delay::default());
        slow.state().delayed_residency = Some(residency.clone());
        let during_residency = async {
            residency.entered().await;
            within(healthy("residency")).await;
            let mut competing = tokio::task::JoinSet::new();
            for _ in 0..4 {
                let req = router.control(operation("query", &slow_task, READ));
                competing.spawn(async move { req.send().await.unwrap().status() });
            }
            assert_eq!(within(competing.join_next()).await.unwrap().unwrap(), 429);
            healthy_inference().await;
            residency.unblock();
        };
        let (view, ()) = tokio::join!(
            router.operation("query", &slow_task, READ),
            during_residency
        );
        assert_eq!(op(&view)["ready_scope"]["end_token"], 4);
    }
}

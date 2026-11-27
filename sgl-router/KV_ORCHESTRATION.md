# Session KV cache orchestration

The optional Router orchestrator lets an application prepare KV for a later
request in the same task or branch. Preparation uses the caller's worker URL
when supplied. Otherwise the Router prefers the session's last routed worker,
then considers healthy workers in load order. It checks prefix compatibility
and the advertised action and transfer path before placing work. Engines execute
commands through HiCache and manage capacity, residency, and source reclamation.
Normal inference remains available when control is disabled, a worker is unavailable,
or preparation is incomplete.

Enable Engine KV control with `--kv-control-config` and hierarchical caching.
Use a shared Mooncake namespace for storage transfers. Configure the Router with
`--kv-orchestration-config` and a protected copy of
[kv-orchestration.example.json](kv-orchestration.example.json). Supply its
`control_bearer` as a Bearer token on control routes. Use the same model
namespace, layout, page size, and compatible storage configuration on source and
destination Engines.

| Action | Initial use |
| --- | --- |
| `sglang.pin` | Set or release a TTL and/or eviction priority for a complete host prefix. |
| `sglang.prefetch` | Prepare a storage prefix in host memory. The Engine reports the actually available range. |
| `sglang.demote` | Preserve a host prefix in storage and reclaim the eligible host range. The result separates these outcomes. |

Applications can submit these actions directly with `POST /v1/kv/control`. A
command carries a task identity and an idempotent message ID. A target object
can select a session, branch, observed request, or compatible prefix. The
default target is the session. Session and branch selectors freeze their scope
at acceptance. Use Engine capabilities to check supported transfer directions.

The Router discovers capabilities with `GET /v1/kv/capabilities` and sends
version `1.0` actions to the Engine's `POST /v1/kv/control`. It sends concrete
`prefix` targets with token IDs and compatibility on submit, and `tokens`
targets on residency reads. PIN cleanup uses the Router task's locally tracked
claims. A PIN command carries its claim ID, revision, and Engine location.

For example, after registering a task and observing its branch, pin its host
prefix on the worker that owns it:

```json
{
  "kind": "submit",
  "task": {"session_id": "session-1", "incarnation": "task-1"},
  "message_id": "keep-branch",
  "action": "sglang.pin",
  "target": {"kind": "branch", "branch_id": "main"},
  "worker_url": "http://worker-a:30000",
  "location": {"tier": "host"},
  "mode": "acquire",
  "claim_id": "keep",
  "revision": 1,
  "ttl_seconds": 3,
  "required_protection": "eviction_exclusion"
}
```

Send lifecycle events to `POST /v1/kv/session/events`:

1. Send `task_started` and observe the actual rendered input via
   `request_observed` or the inference request's `kv_context`. This establishes
   logical association; it does not assert physical residency.
2. On `tool_wait`, supply the intended source worker and `pressure: true`
   to request host-to-storage DEMOTE. Use PIN to protect a hot prefix in L2.
3. Send `prepare` or `resume` when the application wants storage-to-host
   PREFETCH to start. The event queues work without a predicted wait or timer.
   Send inference at its normal time even if the prefix is still being read.
4. Send `task_finished` or `task_cancelled` to release the task's interests.
   An ordinary inference completion leaves the task active.

The Router reserves cleanup capacity at task admission and keeps a bounded FIFO
of retired incarnations. Callers can omit `retirement_generation` throughout a
task's lifecycle. New tasks can then start across FIFO rollovers without a
capabilities lookup. A retired incarnation's replay protection lasts while its
tombstone is in the FIFO.

For replay protection after eviction, read `retirement_generation` from Router
capabilities when starting a new task. Include that value in its task identity
and inference `kv_context` throughout its lifetime, including query, cancel,
and ack. Eviction advances the generation for new tasks. Active tasks keep the
generation they started with.

Use a new monotonic event revision and an event ID for each lifecycle transition.
`query`, `cancel`, and `ack` use the same task and message ID as the original
intent. Cancel detaches only that intent; shared physical work can continue for
other interests. Release a PIN before acknowledging its intent and wait for
quiescence before discarding an operation's receipt.

If discovery confirms a replacement Engine epoch, the old operation reports
`owner_lost: true` and becomes quiescent. Its result retains the last Engine
receipt. Cancel and acknowledge the old interest to reclaim its operation slot.

For a passive association, the Router records caller token IDs or tokens from
the rendered chat template. Its scope reports the original input length in
`original_end` and the page-aligned prefix length in `controlled_end` (up to
65,536 tokens). The Engine receives the full inference input. A live claim
lookup reports PIN protection. A scoped residency read reports PREFETCH
coverage with the Engine's current cache watermark. Inline watermarks in
Engine replies can invalidate earlier readiness while terminal operation
receipts remain available for acknowledgement.

Send `kv_context` on `/v1/responses` to associate the converted chat request.

For request-triggered prefetch, send `kv_context` on a generation request along
with `x-kv-control-authorization: Bearer <control_bearer>`. The Router consumes
this header locally; inference can use a separate `Authorization` credential.
The existing routing policy picks a worker. The Router then offers the
page-aligned reusable input prefix to that worker's control queue and forwards
inference without waiting for preparation. The hint can share a compatible
in-flight HiCache restore. If tokens or queue capacity are unavailable, the
request takes its normal recovery path. A control receipt by itself does not
prove subsequent inference reused the prefix.

See the [Engine control README](../python/sglang/srt/managers/kv_control/README.md)
for configuration and command envelopes.

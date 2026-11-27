# Engine KV cache control

Native KV control is disabled by default. It lets a caller use three versioned
actions:

| Action | Effect |
| --- | --- |
| `sglang.pin` | Set a TTL, eviction priority, or both for existing host KV. |
| `sglang.prefetch` | Read KV from storage (L3) into host memory (L2), keeping the storage copy. |
| `sglang.demote` | Preserve host KV (L2) in storage (L3), then reclaim the eligible host range. |

A router can submit these actions for application task and tool events. The
Engine owns allocation, transfer, and reference checks. Query actual target
coverage and protection state before treating a receipt as proof of readiness.

## Configuration

```bash
python -m sglang.launch_server --model-path MODEL_PATH \
  --enable-hierarchical-cache \
  --hicache-storage-backend mooncake \
  --kv-control-config '{"namespace":"model-weights-v1","engine_id":"worker-a"}'
```

Configure the Mooncake backend separately. Transfers and PIN require a built-in
FULL-attention cache with persistent host HiCache. One worker can use a single
scheduler or a tensor-parallel group on one node. Rank-sharded transfers use
Mooncake FULL MHA with the page_first_direct host layout. File storage also
supports single-rank tests. Discovery reports the available actions and layout.

## Control API

Use `GET /v1/kv/capabilities` for discovery and `POST /v1/kv/control` for commands.
The Python Engine exposes `kv_control(command)` and `async_kv_control(command)`.
The Engine executes concrete token prefixes. The Router resolves session and
branch selectors before submitting control actions.
Use the Engine ID, epoch and namespace returned by discovery in every command:

```json
{
  "kind": "submit",
  "context": {
    "engine_id": "worker-a",
    "engine_epoch": "epoch-from-discovery",
    "namespace": "model-weights-v1",
    "caller_id": "trusted-controller"
  },
  "envelope": {
    "protocol_version": "0.1",
    "message_id": "keep-prefix-1",
    "actions": [{
      "action_id": "keep",
      "action_type": "sglang.pin",
      "action_version": "1.0",
      "payload": {
        "target": {"kind": "tokens", "namespace": "model-weights-v1", "token_ids": [1, 2, 3, 4]},
        "location": {"tier": "host"},
        "mode": "acquire",
        "claim_id": "prefix-1",
        "revision": 1,
        "ttl_seconds": 30,
        "required_protection": "eviction_exclusion"
      }
    }]
  }
}
```

The standalone endpoints use the admin bearer in `Authorization` when an admin
key is configured. Inference retains opaque backend hints through `kv_hints`.
Standalone control returns an `unsupported` receipt for an unrecognized
`sglang.*` action or version.

Use `kind: "tokens"` with a namespace and token IDs, or `kind: "prefix"` with
the compatibility fields from discovery and all tokens through the end of the
prefix. The Engine applies its namespace to the supplied logical cache salt.
PIN requires an already published host copy.

Acquiring or renewing a PIN requires `ttl_seconds`, `priority`, or both.
TTL starts on the Engine after it holds the host pages. The optional signed
32-bit priority changes eviction order without holding a reference:
unranked pages are reclaimed first, then ranked pages from lowest to highest.
Shared pages use the highest active priority. A TTL expiry removes the hold
and keeps any assigned priority. Renewal updates only supplied fields;
release supplies neither. Eviction invalidates the claim before slots can be
reused. The claim view reports its own priority and any remaining hard TTL.

Revisions increase within each `(caller_id, claim_id)` pair. A new claim ID can start at revision 1
even after the retired history fills. The Engine checks replays against active
claims and the most recent `max_records` retired claims (4,096 by default). Once a terminal record
leaves that FIFO, the Engine treats its claim ID as unseen.

Capabilities, submission, query, claim, and residency replies carry the current
cache watermark. Pass `include_claim_id` with a message query to read its live
claim state in the same reply. `cancel` drains worker access before
releasing resources. Acknowledge with `ack` once the operation is quiescent. An
identical message can be replayed within the retention window; a changed payload
with the same identity is rejected. Use `lookup` to find a message by its caller and
message ID. Capacity refusal preserves the inference
request's normal recovery path. DEMOTE can confirm preservation and report
that a source range is blocked by a live GPU copy, reference, or PIN.

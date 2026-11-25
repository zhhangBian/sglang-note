# Host KV cache PIN protection

Native KV control is disabled by default. Enable it to protect existing host KV
with a TTL, an eviction priority, or both.

## Configuration

```bash
python -m sglang.launch_server --model-path MODEL_PATH \
  --enable-hierarchical-cache \
  --enable-kv-cache-control '{"namespace":"model-weights-v1","engine_id":"worker-a"}'
```

PIN requires the built-in FULL-attention cache with persistent host HiCache.
An Engine can use a single scheduler or a tensor-parallel group on one node.
Discovery reports the available protection modes and the compatible prefix layout.

## Control API

Use `GET /v1/kv/capabilities` for discovery and `POST /v1/kv/control` for commands.
The Python Engine exposes `kv_control(command)` and `async_kv_control(command)`.
Use the Engine ID, epoch, and namespace from discovery in each command context.

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

HTTP inference requests carrying version 1.0 PIN actions require the configured
admin credential. Use the admin bearer in `Authorization`
on an admin-key-only server. When inference also requires a separate user key,
send the user bearer in `Authorization` and the admin bearer in
`X-SGLang-Admin-Authorization`. Other action names and versions pass through to
their backend consumers. Standalone control submissions return an `unsupported`
result for each unrecognized `sglang.*` name or version.

Use a target that matches the worker's page size, layout and cache salt. A PIN
requires an already published host copy. An
attached PIN targeting `"request_cached"` binds only after successful request
completion and cache checkpointing. Its scope ends at the last full page within
the committed prefix and the configured token budget (at most 65,536 tokens).

Acquiring or renewing a PIN requires `ttl_seconds`, `priority`, or both.
TTL starts on the Engine after it holds the host pages. The optional signed
32-bit priority changes eviction order without holding a reference. The Engine
reclaims unranked pages first, then ranked pages from lowest to highest.
Shared pages use the highest active priority. A TTL expiry removes the hold
and keeps any assigned priority. Renewal updates only supplied fields.
Omit both fields when releasing a claim. Eviction invalidates the claim before slots can be
reused. The claim view reports its own priority and any remaining hard TTL.

Revisions increase within each claim. A new claim ID can start at revision 1
even after the retired history fills. The Engine checks replays against active
claims and the most recent `max_records` retired claims. Once a terminal record
leaves that FIFO, the Engine treats its claim ID as unseen.

Capabilities, submission, query, claim, and residency replies carry the current
inline `cache_watermark`. Pass `include_claim_id` with a message query to read its
live claim state in the same reply. Use `cancel` to retire a queued acquire and `ack`
once an operation is quiescent. An identical message can be replayed within the
retention window. Reusing the same message identity with a changed payload is
rejected. If a PIN is refused for capacity, inference retains its usual recovery
path.

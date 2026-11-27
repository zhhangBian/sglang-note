# SPDX-License-Identifier: Apache-2.0
"""Typed, scheduler-owned control service; no model request is needed for RPCs."""

from __future__ import annotations

import hashlib
import math
import threading
import time
import uuid
from functools import wraps
from typing import Annotated

import msgspec
from sglang.srt.managers.io_struct import KVControlReqInput, KVControlReqOutput
from sglang.srt.managers.kv_control.contracts import (
    CacheWatermark,
    ControlContext,
    ControlError,
    Identifier,
    Limits,
    Location,
    OperationRef,
    Position,
    TokensTarget,
    Value,
    resolve_prefix,
    validated,
)
from sglang.srt.managers.kv_control.controller import KVControl
from sglang.srt.managers.kv_control.key_namespace import (
    SALT_ENCODING,
    encode_cache_salt,
)
from sglang.srt.managers.kv_hints import KvHintsEnvelope
from sglang.srt.mem_cache.unified_cache.control_protection import (
    HostCacheProtection,
    ProtectionLimits,
)
from sglang.srt.mem_cache.unified_cache.control_state import CacheControlConfig


class CacheLimits(Value):
    max_query_nodes: int = 4096
    max_result_targets: int = 4096
    max_prepared_prefixes: int = 128
    max_concurrent_storage_reads: Annotated[int, msgspec.Meta(ge=1, le=64)] = 4


class ServiceConfig(Value):
    namespace: Identifier
    engine_id: Identifier | None = None
    limits: Limits = Limits()
    cache_limits: CacheLimits = CacheLimits()
    protection_limits: ProtectionLimits = ProtectionLimits()

    def cache_config(self):
        return CacheControlConfig(
            namespace=self.namespace, **msgspec.to_builtins(self.cache_limits)
        )


def parse_config(value):
    if value is None:
        return None
    config = validated(value, ServiceConfig)
    config.cache_config()
    return config


def runtime_config():
    from sglang.srt.runtime_context import (
        get_disagg,
        get_memory,
        get_parallel,
        get_spec,
    )

    config = parse_config(get_memory().kv_control_config)
    if config is None:
        return None
    parallel = get_parallel()
    if (
        any(
            size != 1
            for size in (
                parallel.pp_size,
                parallel.attn_cp_size,
                parallel.attn_dcp_size,
                parallel.num_dp_ranks,
                parallel.nnodes,
            )
        )
        or parallel.attn_dp_enabled
    ):
        raise ValueError(
            "KV control requires one TP group on one node, without PP, CP or DP"
        )
    if get_spec().speculative_algorithm or get_disagg().disaggregation_mode != "null":
        raise ValueError(
            "KV control does not support speculative decoding or PD disaggregation"
        )
    memory = get_memory()
    if (
        memory.disable_radix_cache
        or memory.radix_cache_backend
        or memory.enable_lmcache
        or memory.enable_flexkv
    ):
        raise ValueError("KV control requires the enabled built-in unified radix cache")
    return config


PageSize = Annotated[int, msgspec.Meta(ge=1, le=256)]


class Capabilities(Value, tag="capabilities", tag_field="kind"):
    pass


class Submit(Value, tag="submit", tag_field="kind"):
    context: ControlContext
    envelope: KvHintsEnvelope


class Query(Value, tag="query", tag_field="kind"):
    context: ControlContext
    message_id: Identifier
    include_claim_id: Identifier | None = None


class Lookup(Value, tag="lookup", tag_field="kind"):
    context: ControlContext
    message_id: Identifier
    limit: PageSize = 256


class Ack(Value, tag="ack", tag_field="kind"):
    context: ControlContext
    message_id: Identifier


class Cancel(Value, tag="cancel", tag_field="kind"):
    ref: OperationRef


class Residency(Value, tag="residency", tag_field="kind"):
    context: ControlContext
    target: TokensTarget
    limit: PageSize = 256
    location: Location = Location(tier="gpu")


class Capacity(Value, tag="capacity", tag_field="kind"):
    context: ControlContext


class Claim(Value, tag="claim", tag_field="kind"):
    context: ControlContext
    claim_id: Identifier


class ClaimEvents(Value, tag="claim_events", tag_field="kind"):
    context: ControlContext
    after_sequence: Position = 0
    limit: PageSize = 256


Command = (
    Capabilities
    | Submit
    | Query
    | Lookup
    | Ack
    | Cancel
    | Residency
    | Capacity
    | Claim
    | ClaimEvents
)


class KVControlService:
    def __init__(
        self,
        cache,
        config: ServiceConfig,
        *,
        clock=time.monotonic,
        namespace_keys=False,
        tp_group=None,
    ):
        if type(namespace_keys) is not bool:
            raise ValueError("namespace_keys must be a bool")
        self.namespace_keys = namespace_keys
        config = parse_config(config)
        if config is None:
            raise ValueError("An enabled KV service requires configuration")
        self.cache, self.config = cache, config
        self.tp_group = tp_group
        self.tp_size = 1
        self.tp_time: float | None = None
        engine_id, epoch = config.engine_id or uuid.uuid4().hex, None
        self.key_namespace = config.namespace
        if tp_group is not None:
            import torch.distributed as dist

            self.tp_size = dist.get_world_size(tp_group)
            rank = dist.get_rank(tp_group)
            identity = (
                [engine_id, uuid.uuid4().hex, clock()] if rank == 0 else [None] * 3
            )
            dist.broadcast_object_list(identity, group=tp_group, group_src=0)
            engine_id, epoch, self.tp_time = identity
            clock = self._tp_clock
            # Rank suffixes alone cannot distinguish TP2's first shard from
            # TP4's first shard when both use the same model and Store.
            pool = cache.token_to_kv_pool_allocator.get_kvcache()
            self.key_namespace = (
                f"{config.namespace}:tp{self.tp_size}"
                f":p{cache.tree_core.page_size}:{pool.dtype}"
            )
        if self.namespace_keys:
            self.cache_salt(None)  # Reject an unencodable namespace at startup.
        self.control = KVControl(
            engine_id,
            config.namespace,
            enabled=True,
            limits=config.limits,
            engine_epoch=epoch,
            clock=clock,
        )
        self.control.executor = self._bind_executor()
        capabilities = self.control.executor.capabilities()
        if capabilities["reason"]:
            raise ValueError(
                "KV control cache is unsupported: " + capabilities["reason"]
            )
        cache.kv_control_service = self
        self.control.registry.cache_watermark = self.cache_watermark

    def _tp_clock(self):
        return self.tp_time

    def prepare_shards(self, prepare):
        """Collect every rank's preparation verdict before committing ownership."""
        if self.tp_group is None:
            return prepare()
        import torch.distributed as dist

        result, error = None, None
        try:
            result = prepare()
        except Exception as exc:
            error = (getattr(exc, "code", "shard_admission_failed"), str(exc)[:512])
        verdicts = [None] * self.tp_size
        dist.all_gather_object(verdicts, error, group=self.tp_group)
        for failed in verdicts:
            if failed is not None:
                raise ControlError(*failed)
        return result

    def all_shards_ready(self, ready: bool) -> bool:
        if self.tp_group is None:
            return ready
        import torch
        import torch.distributed as dist

        vote = torch.tensor(int(ready), dtype=torch.int32, device="cpu")
        dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=self.tp_group)
        return bool(vote.item())

    def cache_watermark(self):
        recorder = self.cache.tree_core.kv_events
        return CacheWatermark(
            source_id="cache:"
            + hashlib.sha256(self.control.registry.engine_id.encode()).hexdigest(),
            source_epoch=recorder.state_epoch,
            generation=recorder.state_generation,
            version=recorder.state_version,
        )

    def _bind_executor(self):
        executor = self.cache.create_kv_control_executor(
            engine_id=self.control.registry.engine_id,
            engine_epoch=self.control.registry.engine_epoch,
        )
        if self.namespace_keys:
            executor.cache_salt_transform = self.cache_salt
        if executor.host_control:
            executor.protection = HostCacheProtection(
                executor,
                limits=self.config.protection_limits,
                clock=self.control._clock,
                max_result_targets=self.config.limits.max_scope_spans,
                prepare_shards=self.prepare_shards
                if self.tp_group is not None
                else None,
            )
        return executor

    def cache_salt(self, original):
        return encode_cache_salt(self.key_namespace, original)

    def capabilities(self):
        from sglang.srt.managers.kv_control.public import public_capabilities

        registry = self.control.registry
        executor = self.control.executor.capabilities()
        public = public_capabilities(executor)
        return {
            "enabled": True,
            "public_protocol": public,
            "cache_watermark": self.cache_watermark(),
            "engine_id": registry.engine_id,
            "engine_epoch": registry.engine_epoch,
            "namespace": registry.namespace,
            "executor": {**executor, "actions": tuple(public["actions"])},
            "topology": "tensor_parallel"
            if self.tp_group is not None
            else "single_owner",
            "tp_size": self.tp_size,
            "weight_updates": "unsupported",
            "cache_salt_namespace": self.key_namespace if self.namespace_keys else None,
            "cache_salt_encoding": SALT_ENCODING
            if self.namespace_keys
            else "owner_assertion",
        }

    @property
    def has_pending(self):
        reads = self.control.executor.storage_reads
        return self.control.registry.capacity()["active"] != 0 or (
            reads is not None and bool(reads.jobs or reads.demands)
        )

    @property
    def has_background_work(self):
        protection = self.control.executor.protection
        return self.has_pending or (
            protection is not None and protection.has_background_work
        )

    @property
    def next_deadline(self):
        protection = self.control.executor.protection
        return protection.next_deadline if protection is not None else None

    def invalidate_protection(self, *, reason):
        protection = self.control.executor.protection
        return protection.invalidate(reason=reason) if protection is not None else ()

    def progress_protection(self, *, limit=64):
        protection = self.control.executor.protection
        if protection is not None:
            protection.progress(limit=limit)

    def host_protection_evicted(self, values):
        protection = self.control.executor.protection
        if protection is not None and protection.registry.priorities:
            protection.host_evicted(values)

    def progress(self):
        if self.has_pending:
            self.control.progress(background=True)
        self.progress_protection()

    def reset_after_cache_flush(self):
        epoch = None
        if self.tp_group is not None:
            previous = self.control.registry.engine_epoch
            epoch = hashlib.sha256(f"reset:{previous}".encode()).hexdigest()[:32]
        self.control.reset(self.config.namespace, engine_epoch=epoch)
        self.control.executor = self._bind_executor()
        self.control.registry.cache_watermark = self.cache_watermark

    def _claim_view(self, caller_id, claim_id):
        claim = self.control.executor.protection.get(
            caller_id=caller_id, claim_id=claim_id
        )
        return {
            **msgspec.to_builtins(claim),
            "cache_watermark": self.cache_watermark(),
            "active": claim.state == "active",
            "ttl_remaining_ms": int(1000 * claim.remaining_ttl)
            if claim.remaining_ttl is not None
            else None,
        }

    def execute(self, command):
        if threading.get_ident() != self.cache._kv_control_owner_thread:
            raise ControlError(
                "wrong_owner", "Control commands must run on the cache owner"
            )
        control, registry = self.control, self.control.registry
        if isinstance(command, Capabilities):
            return self.capabilities()
        context = (
            command.ref.context if isinstance(command, Cancel) else command.context
        )
        registry.validate_context(context)
        if isinstance(command, Submit):
            # Independent control has no external consumer. Opaque actions
            # remain in the reply for their owner to handle.
            submission = control.submit(command.envelope, context)
            return {
                **msgspec.to_builtins(submission),
                "cache_watermark": self.cache_watermark(),
            }
        if isinstance(command, Query):
            submission = registry.query(context, command.message_id)
            reply = msgspec.to_builtins(submission)
            protection = control.executor.protection
            if command.include_claim_id is not None and protection is not None:
                try:
                    reply["claim"] = self._claim_view(
                        context.caller_id, command.include_claim_id
                    )
                except ControlError as error:
                    if error.code != "missing":
                        raise
                    reply["claim"] = None
            return {**reply, "cache_watermark": self.cache_watermark()}
        if isinstance(command, Lookup):
            return registry.lookup(context, command.message_id, limit=command.limit)
        if isinstance(command, Ack):
            return registry.acknowledge(context, command.message_id)
        if isinstance(command, Cancel):
            return control.cancel(command.ref)
        if isinstance(command, (Claim, ClaimEvents)):
            protection = control.executor.protection
            if protection is None:
                raise ControlError("unsupported", "L2 protection is unavailable")
            if context.binding_id is not None:
                raise ControlError(
                    "invalid_payload",
                    "Claim queries require an independent caller context",
                )
            if isinstance(command, Claim):
                return self._claim_view(context.caller_id, command.claim_id)
            return protection.changes(
                caller_id=context.caller_id,
                after_sequence=command.after_sequence,
                limit=command.limit,
            )
        if isinstance(command, Residency):
            snapshot = control.executor.residency(
                resolve_prefix(command.target, context, self.config.limits),
                limit=command.limit,
                location=command.location,
            )
            return {
                **msgspec.to_builtins(snapshot),
                "cache_watermark": self.cache_watermark(),
                "coverage_semantics": "requested_scope_only",
            }
        return {
            "operations": registry.capacity(),
            "storage_reads": self.cache.prefetch_outcome_stats_snapshot()
            if control.executor.storage_reads is not None
            else None,
            "protection": {
                "active_claims": control.executor.protection.registry.active_claims,
                "protected_tokens": control.executor.protection.registry.protected_tokens,
            }
            if control.executor.protection is not None
            else None,
        }


def dispatch(service, data):
    """Bounded JSON-compatible reply used by typed IPC and in-process callers."""
    try:
        if len(msgspec.json.encode(data)) > 1 << 20:
            raise ControlError("capacity", "Control request exceeds byte limit")
        command = validated(data, Command)
        if service is None:
            if isinstance(command, Capabilities):
                return {"success": True, "data": {"enabled": False, "actions": []}}
            raise ControlError("unsupported", "Native KV control is disabled")
        return {"success": True, "data": msgspec.to_builtins(service.execute(command))}
    except (ControlError, ValueError, TypeError, RecursionError) as error:
        return {
            "success": False,
            "error": {
                "code": getattr(error, "code", "invalid_payload"),
                "reason": str(error)[:512],
            },
        }


def create_scheduler_service(cache):
    from sglang.srt.runtime_context import get_memory, get_parallel

    if get_memory().kv_control_config is None:
        return None
    parallel = get_parallel()
    return KVControlService(
        cache,
        runtime_config(),
        namespace_keys=True,
        tp_group=parallel.tp_group.cpu_group if parallel.tp_size > 1 else None,
    )


def scheduler_control_handler(service):
    def handle(request):
        return KVControlReqOutput(
            request_id=request.request_id, result=dispatch(service, request.command)
        )

    return handle


def namespace_cache_salt(service, req):
    if service is not None and service.namespace_keys:
        try:
            req.cache_salt = service.cache_salt(req.cache_salt)
        except ControlError as error:
            req.set_finish_with_abort(str(error), status_code=400)


def sleep_on_idle(sleeper, service):
    deadline = service.next_deadline if service is not None else None
    if deadline is None:
        sleeper.maybe_sleep()
    else:
        timeout_ms = max(0, min(1000, math.ceil((deadline - time.monotonic()) * 1000)))
        sleeper.maybe_sleep(timeout_ms=timeout_ms)


def requires_mutable_weights(response_type):
    return _requires_uncontrolled_cache(
        response_type, "KV control requires fixed model weights"
    )


def requires_resident_memory(response_type):
    return _requires_uncontrolled_cache(
        response_type, "KV control requires resident model and cache memory"
    )


def _requires_uncontrolled_cache(response_type, message):
    def decorate(function):
        @wraps(function)
        def wrapped(self, recv_req):
            if getattr(self.scheduler, "kv_control_service", None) is not None:
                return response_type(success=False, message=message)
            return function(self, recv_req)

        return wrapped

    return decorate


async def send_tokenizer_control(manager, command):
    from sglang.srt.runtime_context import get_parallel

    manager.auto_create_handle_loop()
    parallel = get_parallel()
    if parallel.num_dp_ranks != 1 or parallel.pp_size != 1:
        return {
            "success": False,
            "error": {
                "code": "unsupported",
                "reason": "KV control requires one TP group",
            },
        }
    response = await manager.kv_control_communicator(
        KVControlReqInput(request_id=uuid.uuid4().hex, command=command)
    )
    return response[0].result

# SPDX-License-Identifier: Apache-2.0
"""Callable native control core; serving/cache integration supplies Executor."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import msgspec

from sglang.srt.managers.kv_control.contracts import (
    PROTOCOL_VERSION,
    ClaimState,
    ControlContext,
    ControlError,
    ExecutionUpdate,
    Limits,
    OperationRef,
    OperationResult,
    Pin,
    ResolvedTarget,
    Status,
    Submission,
    TokensTarget,
    parse_action,
    resolve_prefix,
)
from sglang.srt.managers.kv_control.public import (
    normalize_public,
    requested_outcomes,
)
from sglang.srt.managers.kv_control.registry import OperationRegistry, snapshot_envelope
from sglang.srt.managers.kv_hints import KvHintsEnvelope


class Executor(Protocol):
    """Scheduler-owned cache protection boundary."""

    def capabilities(self) -> dict: ...

    def pin(
        self, *, ref: OperationRef, payload: Pin, scope: ResolvedTarget | None
    ) -> tuple[ExecutionUpdate, ClaimState | None]: ...


@dataclass
class _Pending:
    ref: OperationRef
    payload: Pin
    target: ResolvedTarget | None


class KVControl:
    """Opt-in, mechanism-only core with no scheduler, policy or backend imports."""

    def __init__(
        self,
        engine_id: str,
        namespace: str,
        *,
        enabled: bool = False,
        executor: Executor | None = None,
        limits: Limits = Limits(),
        engine_epoch: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if type(enabled) is not bool:
            raise ValueError("enabled must be a bool")
        self.enabled = enabled
        self.executor = executor
        self.limits = limits
        self._clock = clock
        self.registry = OperationRegistry(
            engine_id, namespace, engine_epoch=engine_epoch, limits=limits, clock=clock
        )
        self._pending: dict[str, _Pending] = {}
        # Attached actions acquire cache resources in submission order. A
        # hash-dependent order could pick different victims on TP ranks.
        self._requests: dict[tuple[str, str], dict[str, None]] = {}

    def context(
        self, caller_id: str, *, binding_id: str | None = None
    ) -> ControlContext:
        return self.registry.validate_context(
            ControlContext(
                engine_id=self.registry.engine_id,
                engine_epoch=self.registry.engine_epoch,
                namespace=self.registry.namespace,
                caller_id=caller_id,
                binding_id=binding_id,
            )
        )

    def submit(
        self,
        envelope: KvHintsEnvelope,
        context: ControlContext,
        *,
        execute_primitives: bool = True,
    ) -> Submission:
        if type(execute_primitives) is not bool:
            raise ControlError("invalid_payload", "Execution support must be a bool")
        batch, unhandled = snapshot_envelope(
            envelope, self.limits, request_attached=context.binding_id is not None
        )
        if not batch.actions:
            return Submission(replayed=False, unhandled_actions=unhandled)
        submission = self.registry.reserve(batch, context)
        if submission.replayed:
            return msgspec.structs.replace(submission, unhandled_actions=unhandled)
        # Freeze each prefix at submission; action order is not a dependency API.
        prepared = []
        for frozen in batch.actions:
            try:
                if not self.enabled:
                    raise ControlError("unsupported", "Native KV control is disabled")
                if batch.protocol_version != PROTOCOL_VERSION:
                    raise ControlError(
                        "unsupported", "Unsupported native protocol version"
                    )
                payload = parse_action(frozen.thaw())
                if frozen.invalid_numbers:
                    raise ControlError(
                        "invalid_payload",
                        "JSON numbers must be finite and within the exact integer bound",
                    )
                scope = None
                if self.executor is None or not execute_primitives:
                    raise ControlError(
                        "unsupported", "No cache executor for this request"
                    )
                payload = normalize_public(payload, self.executor.capabilities())
                target = payload.target
                if payload.location.engine_id not in (None, context.engine_id):
                    raise ControlError(
                        "unsupported", "Control must address this engine"
                    )
                # Renew/release use the claim identity, even after its prefix is gone.
                if payload.mode == "acquire":
                    if target == "request_cached":
                        if context.binding_id is None:
                            raise ControlError(
                                "invalid_payload",
                                "request_cached requires a request incarnation",
                            )
                    else:
                        scope = resolve_prefix(target, context, self.limits)
                prepared.append((payload, scope))
            except ControlError as error:
                prepared.append(error)
        for result, item in zip(submission.results, prepared):
            if not isinstance(item, ControlError):
                payload, scope = item
                if payload.mode == "acquire" and payload.target == "request_cached":
                    self._pending[result.ref.operation_id] = _Pending(
                        result.ref,
                        payload,
                        scope,
                    )
                    self.registry.set_public_scope(
                        result.ref,
                        requested_outcomes(scope) if scope is not None else (),
                        self.executor.capabilities().get("cache_generation"),
                    )
        for result, item in zip(submission.results, prepared):
            if self.registry.get(result.ref).status.terminal:
                continue
            try:
                if isinstance(item, ControlError):
                    raise item
                payload, scope = item
                self._dispatch(result.ref, payload, scope)
            except ControlError as error:
                self._error(result.ref, error)
        return msgspec.structs.replace(
            self.registry.query(context, batch.message_id), unhandled_actions=unhandled
        )

    def _dispatch(self, ref, payload, scope):
        if payload.target == "request_cached" and payload.mode == "acquire":
            key = ref.context.caller_id, ref.context.binding_id
            self._requests.setdefault(key, {})[ref.operation_id] = None
            self.registry.update(
                ref,
                ExecutionUpdate(
                    status=Status.QUEUED, quiescent=True, code="request_completion"
                ),
            )
        else:
            self._execute_pin(ref, payload, scope)

    def _execute_pin(self, ref, payload, scope):
        if scope is not None:
            self.registry.set_public_scope(
                ref,
                requested_outcomes(scope),
                self.executor.capabilities().get("cache_generation"),
            )
        update, claim = self.executor.pin(ref=ref, payload=payload, scope=scope)
        self._finish(ref, update)
        if claim is not None:
            self.registry.set_public_claim(ref, claim)

    def request_event(
        self,
        context: ControlContext,
        event: Literal["success", "abort", "error", "preempt", "checkpoint"],
        *,
        committed: TokensTarget | None = None,
    ) -> tuple[OperationResult, ...]:
        """success must run before request references are released.

        The owner supplies authoritative committed content for request_cached;
        its executor acquires separate holds synchronously. This core
        never observes a live Req or guesses a generated suffix.
        """
        self.registry.validate_context(context)
        if context.binding_id is None or event not in (
            "success",
            "abort",
            "error",
            "preempt",
            "checkpoint",
        ):
            raise ControlError(
                "invalid_payload", "Expected a request incarnation and lifecycle event"
            )
        if event in ("preempt", "checkpoint"):
            return ()
        key = context.caller_id, context.binding_id
        ids = tuple(self._requests.pop(key, ()))
        results = []
        for operation_id in ids:
            pending = self._pending[operation_id]
            if event != "success":
                results.append(
                    self._finish(
                        pending.ref,
                        ExecutionUpdate(
                            status=Status.CANCELLED,
                            quiescent=True,
                            code="request_failed",
                        ),
                    )
                )
                continue
            try:
                if pending.target is None:
                    if committed is None:
                        raise ControlError(
                            "missing", "No committed request content supplied"
                        )
                    pending.target = resolve_prefix(committed, context, self.limits)
                self._execute_pin(pending.ref, pending.payload, pending.target)
            except ControlError as error:
                self._error(pending.ref, error)
            results.append(self.registry.get(pending.ref))
        return tuple(results)

    def _finish(self, ref, update):
        result = self.registry.update(ref, update)
        if result.status.terminal:
            self._pending.pop(ref.operation_id, None)
            key = ref.context.caller_id, ref.context.binding_id
            ids = self._requests.get(key)
            if ids is not None:
                ids.pop(ref.operation_id, None)
                if not ids:
                    del self._requests[key]
        return result

    def _error(self, ref, error):
        status = (
            Status.UNSUPPORTED
            if error.code in ("unsupported", "unaligned")
            else (Status.MISSING if error.code == "missing" else Status.REJECTED)
        )
        return self._finish(
            ref,
            ExecutionUpdate(
                status=status, quiescent=True, code=error.code, reason=str(error)[:4096]
            ),
        )

    def cancel(self, ref: OperationRef) -> OperationResult:
        current = self.registry.get(ref)  # Includes expected incarnation check.
        if current.status.terminal:
            return current
        return self._finish(
            ref,
            ExecutionUpdate(
                status=Status.CANCELLED,
                quiescent=True,
                code="cancelled_before_dispatch",
            ),
        )

    def progress(self, *, limit: int = 64) -> tuple[OperationResult, ...]:
        if type(limit) is not int or not 1 <= limit <= 4096:
            raise ControlError("invalid_payload", "Progress batch exceeds limit")
        return ()

    def reset(self, namespace: str, *, engine_epoch: str | None = None) -> None:
        """Rotate logical generation after the owner has drained/reset real caches."""
        if self.registry.capacity()["active"]:
            raise ControlError(
                "busy", "Drain active operations before rotating the epoch"
            )
        self.registry = OperationRegistry(
            self.registry.engine_id,
            namespace,
            engine_epoch=engine_epoch,
            limits=self.limits,
            clock=self._clock,
        )

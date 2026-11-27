# SPDX-License-Identifier: Apache-2.0
"""Control protocol, bounded replay and task/request lifecycle."""

import copy
import unittest
from contextlib import contextmanager

import msgspec
import pytest
from sglang.srt.managers.kv_control import KVControl
from sglang.srt.managers.kv_control.contracts import (
    HASH_SCHEME,
    ClaimState,
    ControlError,
    Demote,
    ExecutionUpdate,
    Limits,
    Location,
    Pin,
    Prefetch,
    PublicResult,
    Status,
    TargetOutcome,
    TokenRange,
    TokensTarget,
    parse_action,
    validated,
)
from sglang.srt.managers.kv_control.key_namespace import encode_cache_salt
from sglang.srt.managers.kv_control.public import normalize_public
from sglang.srt.managers.kv_hints import KvHintAction, KvHintsEnvelope
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class Clock:
    now = 100.0

    def __call__(self):
        return self.now


class CompletionBoundary:
    """A controlled external executor, without any pretend cache implementation."""

    def __init__(self):
        self.work = {}
        self.updates = {}
        self.starts = 0

    def capabilities(self):
        return TestPublicKVProtocol.caps()

    def start(self, work):
        self.starts += 1
        self.work[work.ref.operation_id] = work
        return ExecutionUpdate(status=Status.RUNNING, quiescent=False)

    def poll(self, ref):
        return self.updates.pop(ref.operation_id, None)

    def cancel(self, ref):
        return None


def action(kind="prefetch", payload=None, *, action_id="a", version="1.0"):
    return KvHintAction(
        action_id=action_id,
        action_type=f"sglang.{kind}",
        action_version=version,
        payload=payload if payload is not None else movement(),
    )


def envelope(*actions, message_id="m", protocol="0.1"):
    return KvHintsEnvelope(
        protocol_version=protocol, message_id=message_id, actions=list(actions)
    )


def tokens(values=(1, 2, 3, 4), **kwargs):
    return TokensTarget(namespace="ns", token_ids=tuple(values), **kwargs)


def movement(target=None, *, src="storage", dst="host"):
    return {
        "target": wire(target if target is not None else tokens()),
        "source": {"tier": src},
        "destination": {"tier": dst},
    }


def submit(core, *actions, message_id="m", context=None, protocol="0.1"):
    return core.submit(
        envelope(*actions, message_id=message_id, protocol=protocol),
        context or core.context("router"),
    )


def wire(value):
    return msgspec.json.decode(msgspec.json.encode(value))


@contextmanager
def fails(code):
    with pytest.raises(ControlError) as caught:
        yield
    assert caught.value.code == code


def completed_outcome(
    *,
    branch=0,
    component="full",
    content_id="h",
    end=4,
    tier="host",
    steps=("destination_ready",),
):
    return TargetOutcome(
        branch_index=branch,
        coverage=TokenRange(component=component, start_token=0, end_token=end),
        content_id=content_id,
        location=Location(tier=tier),
        status=Status.COMPLETED,
        completed_steps=steps,
    )


class TestKVControl(CustomTestCase):
    def test_mixed_owners_errors_and_replay_are_independent(self):
        core = KVControl("engine", "ns", enabled=True, executor=CompletionBoundary())
        opaque_payload = {"backend_owns": object()}
        opaque = KvHintAction(
            action_id="a",
            action_type="backend.custom",
            action_version="9",
            payload=opaque_payload,
        )
        invalid = action(
            "prefetch", movement() | {"destination": {"tier": True}}, action_id="bad"
        )
        unknown = action("future_action", {}, action_id="unsupported")
        legacy = action("copy_to", movement(), action_id="legacy")
        future_version = action("pin", {}, action_id="new-version", version="2.0")
        original = envelope(opaque, invalid, unknown, legacy, future_version, action())
        result = core.submit(original, core.context("router"))
        assert [r.status for r in result.results] == [
            Status.REJECTED,
            Status.UNSUPPORTED,
            Status.UNSUPPORTED,
            Status.UNSUPPORTED,
            Status.RUNNING,
        ]
        assert result.unhandled_actions == (opaque,)
        assert result.unhandled_actions[0].payload is opaque_payload
        opaque_payload["new"] = object()
        replay = core.submit(original, core.context("router"))
        assert replay.replayed and replay.results == result.results

    def test_snapshot_detaches_mutable_input_and_preserves_conflict_identity(self):
        boundary = CompletionBoundary()
        core = KVControl("engine", "ns", enabled=True, executor=boundary)
        payload = movement()
        original_payload = copy.deepcopy(payload)
        context = core.context("router")
        result = submit(core, action("prefetch", payload), context=context)
        ref = result.results[0].ref
        payload["target"]["token_ids"][0] = 999
        with fails("conflict"):
            submit(core, action("prefetch", payload), context=context)
        assert submit(
            core, action("prefetch", original_payload), context=context
        ).replayed
        assert boundary.work[ref.operation_id].target.content.token_ids == (
            1,
            2,
            3,
            4,
        )
        assert core.registry.get(ref).status == Status.RUNNING
        assert boundary.starts == 1

    def test_context_separates_callers_and_rejects_stale_generations(self):
        core = KVControl("engine", "ns", enabled=True)
        contexts = [core.context(caller) for caller in ("router", "other-router")]
        refs = [submit(core, action(), context=ctx).results[0].ref for ctx in contexts]
        assert len({ref.operation_id for ref in refs}) == 2
        assert all(submit(core, action(), context=ctx).replayed for ctx in contexts)
        for field, value, code in [
            ("engine_id", "other", "wrong_engine"),
            ("engine_epoch", "old", "stale_epoch"),
            ("namespace", "weights-v2", "incompatible_namespace"),
        ]:
            with fails(code):
                submit(
                    core,
                    action(),
                    context=msgspec.structs.replace(contexts[0], **{field: value}),
                )
        assert core.registry.capacity()["records"] == 2

    def test_preflight_limits_do_not_partially_accept_or_mutate(self):
        for limits, actions in [
            (Limits(max_actions=1), [action(action_id="1"), action(action_id="2")]),
            (Limits(max_active=1), [action(action_id="1"), action(action_id="2")]),
        ]:
            with self.subTest(limits=limits, actions=actions):
                core = KVControl("engine", "ns", enabled=True, limits=limits)
                with fails("capacity"):
                    submit(core, *actions)
                assert core.registry.capacity() == {
                    "active": 0,
                    "records": 0,
                    "messages": 0,
                }

    def test_ack_retains_failure_and_replay_deadline_without_renewal(self):
        clock = Clock()
        core = KVControl(
            "engine",
            "ns",
            enabled=True,
            limits=Limits(retention_seconds=10),
            clock=clock,
        )
        context = core.context("router")
        original = submit(core, action("prefetch", movement()), context=context)
        assert original.results[0].status == Status.UNSUPPORTED
        deadline = original.replay_expires_at
        clock.now += 4
        ack = core.registry.acknowledge(context, "m")
        assert ack.summaries[0].status == Status.UNSUPPORTED
        assert ack.summaries[0].reason == original.results[0].update.reason
        assert ack.replay_expires_at == deadline == 110
        assert core.registry.capacity() == {"active": 0, "records": 0, "messages": 1}
        replay = submit(core, action("prefetch", movement()), context=context)
        assert (
            replay.replayed
            and replay.acknowledged
            and (replay.summaries == ack.summaries)
        )
        with fails("conflict"):
            submit(core, action("prefetch", movement(tokens((7, 8, 9, 10)))))
        clock.now = 109.999
        assert core.registry.acknowledge(context, "m") == ack
        clock.now = 110
        new = submit(core, action("prefetch", movement()), context=context)
        assert not new.replayed
        assert new.results[0].ref.operation_id != original.results[0].ref.operation_id

    def test_executor_failure_origin_survives_drain_ack_and_replay(self):
        """A quiescent cancellation receipt must not relabel a backend exception."""
        observed = completed_outcome()

        class FailingBoundary(CompletionBoundary):
            def start(self, work):
                super().start(work)
                return ExecutionUpdate(
                    status=Status.RUNNING,
                    quiescent=False,
                    completed_steps=("destination_ready",),
                    target_outcomes=(observed,),
                )

            def poll(self, ref):
                update = self.updates.pop(ref.operation_id, None)
                if isinstance(update, Exception):
                    raise update
                return update

            def cancel(self, ref):
                raise RuntimeError("secondary drain failure")

        boundary = FailingBoundary()
        core = KVControl("engine", "ns", enabled=True, executor=boundary)
        payload = movement()
        original = submit(core, action("prefetch", payload)).results[0]
        boundary.updates[original.ref.operation_id] = RuntimeError(
            "backend poll failure"
        )
        core.progress()
        origin = "RuntimeError: backend poll failure"
        assert core.registry.get(original.ref).update.reason == origin
        core.progress()
        draining = core.registry.get(original.ref)
        assert draining.status == Status.CANCELLING and (not draining.update.quiescent)
        assert draining.update.reason == origin
        assert core.registry.capacity()["active"] == 1
        with fails("busy"):
            core.registry.acknowledge(core.context("router"), "m")
        boundary.updates[original.ref.operation_id] = ExecutionUpdate(
            status=Status.CANCELLED, quiescent=True
        )
        terminal = core.progress()[0]
        assert terminal.status == Status.FAILED and terminal.update.quiescent
        assert (
            terminal.update.code == "executor_error"
            and terminal.update.reason == origin
        )
        assert terminal.update.completed_steps == ("destination_ready",)
        assert terminal.update.target_outcomes == (observed,)
        assert core.registry.capacity()["active"] == 0
        assert submit(core, action("prefetch", payload)).results == (terminal,)
        ack = core.registry.acknowledge(core.context("router"), "m")
        assert ack.summaries[0].status == Status.FAILED
        assert (
            ack.summaries[0].code == "executor_error"
            and ack.summaries[0].reason == origin
        )
        replay = submit(core, action("prefetch", payload))
        assert (
            replay.replayed
            and replay.acknowledged
            and (replay.summaries == ack.summaries)
        )


class TestPublicKVProtocol(CustomTestCase):
    @staticmethod
    def public_action():
        prefix = dict(
            kind="prefix",
            compatibility=dict(
                namespace="ns",
                cache_salt=None,
                layout_id="python:full:p4:w0",
                page_size=4,
                components=["full"],
                hash_scheme=HASH_SCHEME,
            ),
            start_token=0,
            end_token=4,
            token_ids=[1, 2, 3, 4],
        )
        return action(payload=movement(prefix))

    def test_inline_wire_vectors(self):
        prefetch = self.public_action().payload
        prefix = prefetch["target"]
        demote = movement(prefix, src="host", dst="storage")
        pin = dict(
            target=prefix,
            location={"tier": "host"},
            mode="acquire",
            claim_id="keep",
            revision=1,
            ttl_seconds=10,
            priority=-3,
        )
        vectors = (
            ("prefetch", prefetch, Prefetch),
            ("prefetch", movement(), Prefetch),
            ("demote", demote, Demote),
            ("pin", pin, Pin),
            ("pin", dict(pin, revision=1 << 53), "invalid_payload"),
            ("pin", dict(pin, priority=1 << 31), "invalid_payload"),
            ("pin", dict(pin, mode="release"), "invalid_payload"),
            ("pin", dict(pin, owner_task={}), "invalid_payload"),
            ("prefetch", dict(prefetch, target="request_cached"), "invalid_payload"),
            (
                "prefetch",
                dict(prefetch, target=dict(prefix, blocks=[])),
                "invalid_payload",
            ),
            (
                "prefetch",
                dict(prefetch, target=dict(prefix, end_token=3)),
                "invalid_payload",
            ),
            (
                "prefetch",
                dict(prefetch, target=dict(prefix, token_ids=[True] * 4)),
                "invalid_payload",
            ),
            (
                "prefetch",
                dict(prefetch, target=dict(prefix, token_ids=[1] * 3)),
                "invalid_payload",
            ),
            ("demote", dict(demote, source={"tier": "gpu"}), "unsupported"),
            (
                "prefetch",
                dict(prefetch, source={"tier": "storage", "engine_id": "e"}),
                "invalid_payload",
            ),
        )
        for kind, payload, expected in vectors:
            with self.subTest(kind=kind, payload=payload):
                if isinstance(expected, str):
                    with fails(expected):
                        parse_action(action(kind, payload))
                else:
                    self.assertIsInstance(parse_action(action(kind, payload)), expected)

        receipt = dict(owner_epoch="epoch", state_version=1, outcome_sequence=1)
        claim = dict(
            caller_id="router",
            task_incarnation="caller",
            claim_id="keep",
            revision=2,
            state="released",
            owner_epoch="epoch",
            tombstone=True,
        )
        for schema, value in ((PublicResult, receipt), (ClaimState, claim)):
            encoded = wire(validated(value, schema))
            self.assertEqual(wire(validated(encoded, schema)), encoded)
            self.assertEqual({k: encoded[k] for k in value}, value)
        self.assertIsNone(wire(validated(receipt, PublicResult))["claim"])
        self.assertIsNone(wire(validated(claim, ClaimState))["remaining_ttl"])

        salts = (
            (
                "ns",
                None,
                "f06d7376b149ed3ad521a51027b5f1e9f956206090aa2aa70d278a0101c87eed",
            ),
            (
                "ns",
                "",
                "f06d7376b149ed3ad521a51027b5f1e9f956206090aa2aa70d278a0101c87eed",
            ),
            (
                "ns",
                "tenant",
                "dddf3db6cf3c0a983092769f0bc478164da6c2b01469cc218da5fd524bedb2f2",
            ),
            (
                "权重😀",
                '租户\x00"\\\n',
                "6cce2ac5ec408ea8460c4a7c6eff285b4f7126e0ba2024ac8f4047b5e52bf58b",
            ),
            (
                "ns",
                "é",
                "332501b2526ba1f80fadaf43a4d8d6da1a9160b5a9c05062b6ef7b246b2bfaf5",
            ),
            (
                "ns",
                "é",
                "07b718dbdd7012bc7e0073124c5254159f774db56c02452b8bffd851ab5f07c1",
            ),
        )
        for namespace, salt, expected in salts:
            self.assertEqual(encode_cache_salt(namespace, salt), expected)
        for salt, expected in (("raw", "raw"), ("", None)):
            a = self.public_action()
            a.payload["target"]["compatibility"]["cache_salt"] = salt
            self.assertEqual(
                normalize_public(parse_action(a), self.caps()).target.cache_salt,
                expected,
            )

    @staticmethod
    def caps():
        return {
            "namespace": "ns",
            "layout_id": "python:full:p4:w0",
            "components": ("full",),
            "actions": ("sglang.prefetch", "sglang.demote"),
            "cache_generation": 7,
            "directions": {
                "prefetch": (("storage", "host"),),
                "demote": (("host", "storage"),),
                "storage_backend": "file",
                "page_alignment": 4,
            },
        }

    def owner(self):
        executor = CompletionBoundary()
        executor.capabilities = self.caps
        clock = Clock()
        return (
            KVControl("engine", "ns", enabled=True, executor=executor, clock=clock),
            executor,
            clock,
        )

    def test_demote_projection_preservation_is_not_reclamation(self):
        control, executor, _ = self.owner()
        a = self.public_action()
        a = action(
            "demote",
            dict(a.payload, source={"tier": "host"}, destination={"tier": "storage"}),
        )
        ref = control.submit(envelope(a), control.context("A")).results[0].ref
        coverage = TokenRange(start_token=0, end_token=4, component="full")
        outcome = TargetOutcome(
            coverage=coverage,
            status=Status.PARTIAL,
            location=Location(tier="storage"),
            completed_steps=("destination_ready",),
            code="active_request_reference",
        )
        executor.updates[ref.operation_id] = ExecutionUpdate(
            status=Status.PARTIAL,
            quiescent=True,
            completed_steps=("destination_ready",),
            target_outcomes=(outcome,),
        )
        control.progress()
        result = control.registry.get(ref)
        public = result.public_result
        self.assertEqual(public.ready_scope, (outcome,))
        self.assertEqual(public.preserved_scope, (outcome,))
        self.assertEqual(public.blocked_scope, (outcome,))
        self.assertFalse(public.reclaimed_scope)
        self.assertEqual(public.cache_generation, 7)
        self.assertEqual(public.owner_epoch, ref.context.engine_epoch)
        # Existing consumers still receive their original update and steps.
        self.assertEqual(result.update.target_outcomes, (outcome,))


if __name__ == "__main__":
    unittest.main()

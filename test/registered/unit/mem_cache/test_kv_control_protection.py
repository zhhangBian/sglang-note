# SPDX-License-Identifier: Apache-2.0
"""Native host eviction exclusion, shared budgets and split-aware releases.

The CPU fixtures provide real cache pools, controlled DMA and paired host
registration cleanup.
"""

import asyncio
import unittest
from unittest.mock import patch

import kv_control_test_utils as cache_test_utils
import msgspec
import test_kv_control_cache as cache_fixtures
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sglang.srt.managers.kv_control.contracts import (
    ControlError,
)
from sglang.srt.managers.kv_control.http import install_kv_control_routes
from sglang.srt.managers.kv_control.protection import (
    HostPrefix,
    ProtectionKey,
    ProtectionRegistry,
)
from sglang.srt.managers.kv_control.service import (
    ServiceConfig,
    dispatch,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.control_protection import ProtectionLimits
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
FULL = ComponentType.FULL


def claim(caller="A", name="pin"):
    return ProtectionKey(caller_id=caller, claim_id=name)


def prefixes(*values):
    return tuple(HostPrefix(token_ids=tuple(tokens)) for tokens in values)


def registry(owner, **limits):
    case, cache, service, _ = owner
    result = ProtectionRegistry(
        cache,
        engine_id="engine",
        engine_epoch=service.control.registry.engine_epoch,
        namespace="ns",
        clock=case.clock,
        **limits,
    )
    case.addCleanup(result.invalidate)
    return result


def host_copy(owner, tokens=(1, 2, 3, 4)):
    _, cache, _, dma = owner
    node = cache_test_utils.seed(cache, tokens)
    cache_test_utils.backup_prefix(cache, dma, node)
    return node


class TestKVControlProtection(cache_fixtures.KVCacheTestCase):
    def error(self, code, operation, **kwargs):
        with self.assertRaises(ControlError) as caught:
            operation(**kwargs)
        self.assertEqual(caught.exception.code, code)

    def test_tp_reservation_failure_rolls_back_successful_peer(self):
        with cache_test_utils.hicache(self) as owner:
            _, cache, service, _ = owner
            node = host_copy(owner)
            pins = registry(owner, prepare_shards=service.prepare_shards)
            # Exercise both local views of the same two-rank collective.
            for failing_rank in (False, True):
                votes = []

                def gather(verdicts, error, *, group):
                    votes.append(error)
                    if len(votes) == 1:
                        self.assertIsNone(error)
                        return
                    self.assertEqual(
                        node.component_data[FULL].host_lock_ref, int(not failing_rank)
                    )
                    if failing_rank:
                        self.assertEqual(
                            error, ("shard_admission_failed", "reservation failed")
                        )
                    verdicts[:] = [
                        None,
                        ("shard_admission_failed", "reservation failed"),
                    ]

                with (
                    patch.object(service, "tp_group", object()),
                    patch.object(service, "tp_size", 2),
                    patch("torch.distributed.all_gather_object", side_effect=gather),
                    patch.object(
                        cache, "inc_host_lock_ref", wraps=cache.inc_host_lock_ref
                    ) as hold,
                ):
                    if failing_rank:
                        hold.side_effect = RuntimeError("reservation failed")
                    self.error(
                        "shard_admission_failed",
                        pins.acquire,
                        key=claim(),
                        revision=1,
                        prefixes=prefixes((1, 2, 3, 4)),
                        ttl_seconds=10,
                    )
                self.assertEqual(len(votes), 2)
                self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
                self.assertEqual(pins.protected_tokens, 0)
                self.assertEqual(pins.active_claims, 0)
                self.assertIsNone(pins.get(claim()))
            cache.sanity_check()

    def test_overlapping_callers_and_multi_branch_scope_charge_union_after_splits(self):
        with cache_test_utils.hicache(self) as owner:
            _, cache, _, _ = owner
            leaf = host_copy(owner)
            pins = registry(owner, max_protected_tokens=6)
            pins.acquire(
                key=claim(), revision=1, prefixes=prefixes((1, 2, 3, 4)), ttl_seconds=5
            )
            branch = host_copy(owner, (1, 2, 5, 6))
            parent = leaf.parent
            self.assertEqual(parent.component_data[FULL].host_lock_ref, 1)
            self.assertEqual(leaf.component_data[FULL].host_lock_ref, 1)
            shared = pins.acquire(
                key=claim("B"),
                revision=1,
                prefixes=prefixes((1, 2, 3, 4), (1, 2, 5, 6)),
                ttl_seconds=20,
            )
            self.assertEqual(shared.physical_tokens, 6)
            self.assertEqual(pins.protected_tokens, 6)
            self.assertEqual(pins.active_claims, 2)
            # An overlapping branch/ancestor is charged once within a claim too.
            pins.acquire(
                key=claim("C"), revision=1, prefixes=prefixes((1, 2)), ttl_seconds=10
            )
            self.assertEqual(pins.protected_tokens, 6)
            self.clock.now = 106
            self.assertEqual([s.key for s in pins.expire()], [claim()])
            self.assertEqual(leaf.component_data[FULL].host_lock_ref, 1)
            self.assertEqual(parent.component_data[FULL].host_lock_ref, 2)
            self.assertEqual(cache.evict_host(100), 0)
            pins.release(key=claim("B"), revision=2)
            self.assertEqual(pins.protected_tokens, 2)
            self.assertEqual(leaf.component_data[FULL].host_lock_ref, 0)
            self.assertEqual(branch.component_data[FULL].host_lock_ref, 0)
            self.assertEqual(parent.component_data[FULL].host_lock_ref, 1)
            pins.release(key=claim("C"), revision=2)
            self.assertEqual(parent.component_data[FULL].host_lock_ref, 0)
            self.assertEqual(pins.protected_tokens, 0)
            self.assertEqual(cache.evict_host(100), 6)
            cache.sanity_check()

    def test_active_claim_cap_and_per_claim_revisions_across_retired_fifo_rollover(
        self,
    ):
        with cache_test_utils.hicache(self) as owner:
            held = host_copy(owner)
            extra = host_copy(owner, (8, 9))
            pins = registry(
                owner, max_protected_tokens=4, max_claims=2, max_records=4096
            )
            scope = prefixes((1, 2, 3, 4))
            pins.acquire(key=claim(), revision=1, prefixes=scope, ttl_seconds=10)
            self.error(
                "capacity",
                pins.acquire,
                key=claim("B"),
                revision=1,
                prefixes=prefixes((8, 9)),
                ttl_seconds=10,
            )
            self.assertEqual(extra.component_data[FULL].host_lock_ref, 0)
            pins.acquire(key=claim("B"), revision=1, prefixes=scope, ttl_seconds=10)
            self.error(
                "capacity",
                pins.acquire,
                key=claim("C"),
                revision=1,
                prefixes=scope,
                ttl_seconds=10,
            )
            # Failed revision updates must leave the old scope/deadline intact.
            self.error(
                "capacity",
                pins.acquire,
                key=claim(),
                revision=2,
                prefixes=prefixes((8, 9)),
                ttl_seconds=10,
            )
            self.assertEqual(pins.get(claim()).revision, 1)
            self.assertEqual(held.component_data[FULL].host_lock_ref, 2)
            pins.release(key=claim(), revision=2)
            pins.release(key=claim("B"), revision=2)
            pins.release(key=claim("C"), revision=1)
            self.error("released", pins.renew, key=claim(), revision=3, ttl_seconds=10)
            for i in range(2_000_000):
                key = claim(name=f"pin-{i}")
                ended = pins.release(key=key, revision=2)
            self.assertEqual(len(pins._retired), pins.max_records)
            self.assertIsNone(pins.get(claim(name="pin-0")))
            self.assertIsNotNone(
                pins.get(claim(name=f"pin-{2_000_000 - pins.max_records}"))
            )
            self.error(
                "stale_revision",
                pins.acquire,
                key=key,
                revision=1,
                prefixes=scope,
                ttl_seconds=10,
            )
            self.assertEqual(pins.release(key=key, revision=2), ended)
            self.assertEqual(pins.protected_tokens, 0)
            self.assertEqual(held.component_data[FULL].host_lock_ref, 0)
            owner[1].sanity_check()

    def test_pending_host_backup_cannot_be_claimed_before_publication(self):
        with cache_test_utils.hicache(self) as owner:
            _, cache, service, dma = owner
            node = cache_test_utils.seed(cache)
            cache_test_utils.start_inference_backup(cache, node)
            self.assertIsNotNone(node.component_data[FULL].host_value)
            self.assertIsNotNone(node.write_through_pending_id)
            pins = registry(owner)
            self.error(
                "incomplete",
                pins.acquire,
                key=claim(),
                revision=1,
                prefixes=prefixes((1, 2)),
                ttl_seconds=10,
            )
            self.assertIs(node.parent, cache.tree_core.root_node)
            self.assertEqual(pins.protected_tokens, 0)
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            self.assertGreater(node.component_data[FULL].lock_ref, 0)
            self.clock.now = 105
            cache_test_utils.finish_inference_backup(cache, dma)
            active = pins.acquire(
                key=claim(),
                revision=1,
                prefixes=prefixes((1, 2, 3, 4)),
                ttl_seconds=10,
            )
            self.assertEqual(active.activated_at, 105)
            self.assertEqual(active.expires_at, 115)
            pins.release(key=claim(), revision=2)
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            cache.sanity_check()


def pin_command(
    service,
    *,
    caller="A",
    message="pin",
    mode="acquire",
    revision=1,
    claim_id="pin",
    ttl=10,
    priority=None,
    selection=None,
    salt=None,
):
    caps = service.capabilities()["public_protocol"]["prefix"]
    payload = {
        "mode": mode,
        "target": selection
        or {
            "kind": "prefix",
            "compatibility": {
                "namespace": caps["namespace"],
                "cache_salt": salt,
                "layout_id": caps["layout_id"],
                "page_size": caps["page_size"],
                "components": list(caps["components"]),
                "hash_scheme": caps["hash_scheme"],
            },
            "start_token": 0,
            "end_token": 4,
            "token_ids": [1, 2, 3, 4],
        },
        "location": {"tier": "host"},
        "claim_id": claim_id,
        "revision": revision,
    }
    if mode != "release":
        payload["ttl_seconds"] = ttl
    if priority is not None:
        payload["priority"] = priority
    return {
        "kind": "submit",
        "context": msgspec.to_builtins(service.control.context(caller)),
        "envelope": {
            "protocol_version": "0.1",
            "message_id": message,
            "actions": [
                {
                    "action_id": "a",
                    "action_type": "sglang.pin",
                    "action_version": "1.0",
                    "payload": payload,
                }
            ],
        },
    }


def run_pin(service, **kwargs):
    reply = dispatch(
        service,
        msgspec.json.decode(msgspec.json.encode(pin_command(service, **kwargs))),
    )
    assert reply["success"], reply
    return reply["data"]["results"][0]


def query_claim(service, *, caller="A", claim_id="pin"):
    return dispatch(
        service,
        {
            "kind": "claim",
            "context": msgspec.to_builtins(service.control.context(caller)),
            "claim_id": claim_id,
        },
    )


class TestKVControlPinRuntime(cache_fixtures.KVCacheTestCase):
    def test_http_pin_lifetime_query_and_same_message_replay(self):
        with cache_test_utils.hicache(self) as owner:
            _, cache, service, _ = owner
            node = host_copy(owner)

            class Manager:
                async def kv_control(self, command):
                    return dispatch(service, command)

            app = FastAPI()
            install_kv_control_routes(app, lambda: Manager())

            async def exercise():
                async with AsyncClient(
                    transport=ASGITransport(app), base_url="http://owner"
                ) as client:

                    async def post(command):
                        response = await client.post("/v1/kv/control", json=command)
                        response.raise_for_status()
                        return response.json()["data"]

                    discovery = (await client.get("/v1/kv/capabilities")).json()["data"]
                    mark = msgspec.to_builtins(service.cache_watermark())
                    assert discovery["cache_watermark"] == mark
                    assert (
                        discovery["engine_epoch"]
                        == service.control.registry.engine_epoch
                    )
                    assert discovery["public_protocol"]["actions"]["sglang.pin"] == [
                        "1.0"
                    ]
                    request = pin_command(service)
                    reply = await post(request)
                    result = reply["results"][0]
                    assert reply["cache_watermark"] == mark
                    assert result["update"]["status"] == "completed"
                    assert result["public_result"]["claim"]["state"] == "active"
                    assert not service.has_pending and service.has_background_work
                    self.clock.now = 103
                    replay = await post(request)
                    assert replay["replayed"] and replay["results"] == [result]
                    context = request["context"]
                    claim_query = dict(kind="claim", context=context, claim_id="pin")
                    state = await post(claim_query)
                    assert state["active"] and state["cache_watermark"] == mark
                    assert (
                        state["remaining_ttl"] == 7
                        and state["ttl_remaining_ms"] == 7000
                    )
                    assert state["actual_scope"][0]["coverage"]["end_token"] == 4
                    query = dict(kind="query", context=context, message_id="pin")
                    for claim_id, expected in (("pin", state), ("absent", None)):
                        combined = await post(dict(query, include_claim_id=claim_id))
                        assert combined["claim"] == expected
                        assert combined["cache_watermark"] == mark
                        assert combined["results"] == [result]
                    assert (await post(dict(query, kind="lookup")))[0]["results"] == [
                        result
                    ]
                    renewed = await post(
                        pin_command(
                            service, mode="renew", revision=2, message="renew", ttl=20
                        )
                    )
                    assert (
                        renewed["results"][0]["public_result"]["claim"]["remaining_ttl"]
                        == 20
                    )
                    assert service.next_deadline == 123
                    await post(
                        pin_command(
                            service, mode="release", revision=3, message="release"
                        )
                    )
                    terminal = await post(claim_query)
                    assert terminal["state"] == "released" and terminal["tombstone"]
                    assert terminal["remaining_ttl"] is None
                    for revision, code in ((2, "stale_revision"), (4, "released")):
                        late = await post(
                            pin_command(
                                service,
                                mode="renew",
                                revision=revision,
                                message=f"late-{revision}",
                            )
                        )
                        assert late["results"][0]["update"]["code"] == code

            asyncio.run(exercise())
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            cache.sanity_check()

    def test_priority_orders_host_leaves_and_gpu_duplicates_without_holds(self):
        with cache_test_utils.hicache(self) as owner:
            _, cache, service, dma = owner
            high = host_copy(owner)
            low = cache_test_utils.seed(cache, (5, 6, 7, 8))
            ordinary = cache_test_utils.seed(cache, (9, 10, 11, 12))
            for node in (low, ordinary):
                cache_test_utils.backup_prefix(cache, dma, node, evict=True)
            self.assertIn(high.id, cache.tree_core.full_host_duplicates)

            pin = run_pin(service, ttl=None, priority=0)
            state = pin["public_result"]["claim"]
            self.assertEqual(state["effective_protection"], "eviction_priority")
            self.assertEqual(state["priority"], 0)
            self.assertIsNone(state["remaining_ttl"])
            run_pin(
                service,
                message="low",
                claim_id="low",
                ttl=None,
                priority=-5,
                selection=msgspec.to_builtins(cache_fixtures.target((5, 6, 7, 8))),
            )
            protection = service.control.executor.protection
            self.assertEqual(protection.registry.protected_tokens, 0)
            self.assertIsNone(protection.next_deadline)

            for evicted in (ordinary, low, high):
                self.assertEqual(cache.evict_host(4), 4)
                self.assertIsNone(evicted.component_data[FULL].host_value)
            self.assertIsNotNone(high.component_data[FULL].value)
            self.assertEqual(query_claim(service)["data"]["state"], "invalidated")
            self.assertFalse(protection.registry.priorities)
            cache.sanity_check()

    def test_shared_priority_survives_ttl_split_and_never_follows_recycled_slots(self):
        with cache_test_utils.hicache(self) as owner:
            _, cache, service, dma = owner
            leaf = cache_test_utils.seed(cache)
            alternate = cache_test_utils.seed(cache, (9, 10, 11, 12))
            for node in (leaf, alternate):
                cache_test_utils.backup_prefix(cache, dma, node, evict=True)
            old_slots = set(leaf.component_data[FULL].host_value.tolist())
            run_pin(service, ttl=2, priority=0)
            run_pin(service, caller="B", message="b", ttl=None, priority=7)
            run_pin(
                service,
                message="other",
                claim_id="other",
                ttl=None,
                priority=3,
                selection=msgspec.to_builtins(cache_fixtures.target((9, 10, 11, 12))),
            )
            cache_fixtures.match_len(cache, (1, 2))
            parent = leaf.parent
            self.assertIsNot(parent, cache.tree_core.root_node)
            self.assertEqual(parent.component_data[FULL].host_lock_ref, 1)
            updated = run_pin(
                service,
                mode="renew",
                message="priority",
                revision=2,
                ttl=None,
                priority=1,
            )["public_result"]["claim"]
            self.assertEqual(updated["remaining_ttl"], 2)
            self.clock.now = 102
            # No service tick: capacity eviction itself expires native holds.
            self.assertEqual(cache.evict_host(4), 4)
            self.assertIsNone(alternate.component_data[FULL].host_value)
            self.assertIsNotNone(leaf.component_data[FULL].host_value)
            self.assertEqual(parent.component_data[FULL].host_lock_ref, 0)
            self.assertEqual(leaf.component_data[FULL].host_lock_ref, 0)
            state = query_claim(service)["data"]
            self.assertEqual((state["state"], state["priority"]), ("active", 1))
            self.assertIsNone(state["remaining_ttl"])
            run_pin(
                service, caller="B", mode="release", revision=2, message="release-b"
            )

            # A TTL renewal keeps the supplied priority and reestablishes the
            # native holds on both segments of the split.
            renewed = run_pin(service, mode="renew", message="ttl", revision=3, ttl=2)
            self.assertEqual(renewed["public_result"]["claim"]["priority"], 1)
            self.assertEqual(cache.evict_host(4), 0)
            pool = cache.host_pool_group.anchor_entry.host_pool
            reserved = pool.alloc(pool.available_size())
            self.clock.now = 104
            self.assertEqual(cache.evict_host(4), 4)
            self.assertEqual(query_claim(service)["data"]["state"], "invalidated")
            fresh = host_copy(owner, (13, 14, 15, 16))
            self.assertEqual(
                set(fresh.component_data[FULL].host_value.tolist()), old_slots
            )
            self.assertEqual(cache.evict_host(4), 4)
            pool.free(reserved)
            cache.sanity_check()

    def test_idle_expiry_publishes_after_operation_completion_without_early_scans(self):
        with cache_test_utils.hicache(self) as owner:
            _, cache, service, _ = owner
            node = host_copy(owner)
            run_pin(service, ttl=2)
            protection = service.control.executor.protection
            with patch.object(
                protection.registry,
                "expire",
                side_effect=AssertionError("early full scan"),
            ):
                for _ in range(10):
                    service.progress()
            self.clock.now = 102
            service.progress()
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            events = dispatch(
                service,
                {
                    "kind": "claim_events",
                    "context": msgspec.to_builtins(service.control.context("A")),
                },
            )
            self.assertEqual(
                [e["claim"]["state"] for e in events["data"]["items"]],
                ["active", "expired"],
            )
            self.assertFalse(service.has_background_work)
            self.assertIsNone(service.next_deadline)
            self.assertEqual(query_claim(service)["data"]["state"], "expired")
            cache.sanity_check()

    def test_reset_invalidates_claims_before_host_slots_are_reused(self):
        with cache_test_utils.hicache(self, namespace_keys=True) as owner:
            _, cache, service, _ = owner
            node = host_copy(owner)
            run_pin(service)
            old_epoch = service.control.registry.engine_epoch
            previous = service.control.executor.protection
            cache.reset()
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            self.assertEqual(
                previous.get(caller_id="A", claim_id="pin").state,
                "invalidated",
            )
            service.reset_after_cache_flush()
            self.assertNotEqual(service.control.registry.engine_epoch, old_epoch)
            self.assertFalse(service.has_background_work)
            self.assertEqual(query_claim(service)["error"]["code"], "missing")

    def test_raw_salt_is_encoded_once_and_storage_detach_preserves_l2_claims(self):
        with cache_test_utils.hicache(self, namespace_keys=True) as owner:
            _, cache, service, dma = owner
            node = cache_fixtures.insert(
                cache, (1, 2, 3, 4), salt=service.cache_salt("raw")
            )
            cache_test_utils.backup_prefix(cache, dma, node)
            pinned = run_pin(service, salt="raw")
            self.assertEqual(pinned["public_result"]["claim"]["state"], "active")
            ok, reason = cache.detach_storage_backend()
            self.assertTrue(ok, reason)
            service.capabilities()
            self.assertEqual(query_claim(service)["data"]["state"], "active")
            self.assertEqual(node.component_data[FULL].host_lock_ref, 1)
            run_pin(service, mode="release", revision=2, message="release", salt="raw")
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            cache.sanity_check()

    def test_idle_sleep_timeout_stops_at_protection_deadline(self):
        from types import SimpleNamespace

        from sglang.srt.managers.scheduler import Scheduler

        with cache_test_utils.hicache(self) as owner:
            _, _, service, _ = owner
            host_copy(owner)
            run_pin(service, ttl=0.25)
            sleeps = []
            scheduler = SimpleNamespace(
                kv_control_service=service,
                idle_sleeper=SimpleNamespace(
                    maybe_sleep=lambda **kwargs: sleeps.append(kwargs["timeout_ms"])
                ),
            )
            with patch(
                "sglang.srt.managers.scheduler.time.monotonic", return_value=100
            ):
                Scheduler.maybe_sleep_on_idle(scheduler)
            self.assertEqual(sleeps, [250])
            self.assertFalse(service.has_pending)
            self.clock.now = 101
            service.progress()
            self.assertEqual(query_claim(service)["data"]["state"], "expired")

    def test_bounded_claim_events_signal_gap_and_filter_other_callers(self):
        config = ServiceConfig(
            namespace="ns",
            engine_id="engine",
            protection_limits=ProtectionLimits(max_events=2),
        )
        with cache_test_utils.hicache(self, config=config) as owner:
            _, _, service, _ = owner
            host_copy(owner)
            run_pin(service)
            run_pin(service, mode="renew", revision=2, message="renew")
            run_pin(service, caller="B", message="b")
            reply = dispatch(
                service,
                {
                    "kind": "claim_events",
                    "context": msgspec.to_builtins(service.control.context("A")),
                },
            )["data"]
            self.assertTrue(reply["gap"])
            self.assertEqual([e["claim"]["caller_id"] for e in reply["items"]], ["A"])
            self.assertEqual(reply["next_sequence"], 3)
            self.assertEqual(query_claim(service)["data"]["state"], "active")
            run_pin(service, mode="release", revision=3, message="release")
            run_pin(
                service, caller="B", mode="release", revision=2, message="b-release"
            )


if __name__ == "__main__":
    unittest.main()

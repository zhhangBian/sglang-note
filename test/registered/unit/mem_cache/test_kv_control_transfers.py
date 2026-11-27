# SPDX-License-Identifier: Apache-2.0
"""KV copy/move with real CPU pools and file storage; controlled DMA fences."""

import threading
import unittest
from unittest.mock import patch

import msgspec
import test_kv_control_cache as cache_fixtures
import torch
from kv_control_test_utils import (
    FULL,
    assert_retired,
    backup_prefix,
    block_io,
    cold_prefix,
    current,
    hicache,
    persist_prefix,
    prefix_path,
    public_transfer,
    seed,
    start_inference_load,
    transfer,
    wait,
    wait_for,
)
from sglang.srt.managers.kv_control.contracts import (
    Limits,
    Status,
)
from sglang.srt.managers.kv_control.service import (
    CacheLimits,
    ServiceConfig,
)
from sglang.srt.managers.kv_hints import KvHintAction, KvHintsEnvelope
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    CacheRequestOutcome,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestKVControlTransfers(cache_fixtures.KVCacheTestCase):
    def test_gpu_transfer_directions_are_rejected_and_inference_loads_host_bytes(self):
        with hicache(self, storage=True) as (_, cache, service, dma):
            node = seed(cache)
            for kind, source, destination in (
                ("prefetch", "host", "gpu"),
                ("prefetch", "storage", "gpu"),
                ("demote", "gpu", "host"),
                ("demote", "gpu", "storage"),
            ):
                result = public_transfer(
                    service, kind, source, destination, f"{source}-{destination}"
                )
                self.assertEqual(result.status, Status.UNSUPPORTED)
            self.assertFalse(dma.pending)
            backup_prefix(cache, dma, node, evict=True)
            cd = node.component_data[FULL]
            self.assertIsNone(cd.value)
            pool = cache.token_to_kv_pool_allocator.get_kvcache()
            pool.k_buffer[0].fill_(-7)
            pool.v_buffer[0].fill_(-8)
            start_inference_load(cache, node)
            dma.finish()
            cache.loading_check()
            self.assertTrue(torch.all(pool.k_buffer[0][cd.value] == 7))
            self.assertTrue(torch.all(pool.v_buffer[0][cd.value] == 8))
            self.assertEqual(cd.lock_ref, 0)
            self.assertEqual(cd.host_lock_ref, 0)
            cache.sanity_check()


class TestKVControlStorageReads(cache_fixtures.KVCacheTestCase):
    def _cold_scopes(self, owner, scopes):
        _, cache, service, dma = owner
        for index, target in enumerate(scopes):
            node = seed(
                cache, target.token_ids, value=7 + index * 2, salt=target.cache_salt
            )
            persist_prefix(cache, service, dma, node)
            self.assertEqual(
                cache.evict_host(len(target.token_ids)), len(target.token_ids)
            )

    def _restore(self, service, target, message, *, dst="host"):
        return transfer(
            service,
            src="storage",
            dst=dst,
            selection=target,
            message=message,
        )

    def _ordinary_prefetch(self, cache, service, target, request):
        cache.prefetch_from_storage(
            request,
            cache.tree_core.root_node.id,
            cache_fixtures.array("q", target.token_ids),
            matched_prefix_tokens=[],
            cache_salt=service.cache_salt(target.cache_salt),
        )

    def _consume_ordinary_prefetch(self, cache, service, request):
        def ready():
            cache.drain_storage_control_queues()
            return cache.check_prefetch_progress(request)

        wait_for(service, ready)
        span = cache.pop_prefetch_loaded_span(request)
        cache.discard_storage_prefetch_accounting(request)
        service.progress()
        return span

    def test_queued_reads_park_without_evicting_resident_host_pages(self):
        with hicache(self, namespace_keys=True) as owner:
            _, cache, service, dma = owner
            scopes = [cache_fixtures.target(range(i, i + 24)) for i in (1, 25, 49)]
            self._cold_scopes(owner, scopes)
            hot = seed(cache, range(129, 193))
            backup_prefix(cache, dma, hot, evict=True)
            controller = cache.cache_controller
            entered, release = threading.Event(), threading.Event()
            reads = service.control.executor.storage_reads
            jobs = []

            def submit(index):
                result = self._restore(service, scopes[index], f"native-{index}")
                job = reads.interests[result.ref.operation_id].job
                jobs.append((result, job))
                return result, job

            try:
                with patch.object(
                    controller,
                    "page_get_func",
                    block_io(controller.page_get_func, entered, release),
                ):
                    submit(0)
                    wait_for(service, entered.is_set)
                    submit(1)
                    last, parked = submit(2)
                    wait_for(
                        service,
                        lambda: (
                            getattr(parked.operation, "host_allocation_deferred", None)
                            == "io"
                        ),
                    )
                    assert hot.component_data[FULL].host_value is not None
                    assert cache.ongoing_prefetch[parked.handle].host_indices is None
                    assert controller.prefetch_buffer.qsize() == 1
                    cache.drain_storage_control_queues_local()
                    assert cache.ongoing_prefetch[parked.handle].host_indices is None
                    service.control.cancel(last.ref)
                    assert wait(service, last).status == Status.CANCELLED
                    assert (
                        not release.is_set()
                        and parked.handle not in cache.ongoing_prefetch
                    )
                    release.set()
                    for result, _ in jobs[:2]:
                        assert wait(service, result).status == Status.COMPLETED
            finally:
                release.set()
                for result, _ in jobs:
                    wait(service, result)
            assert_retired(cache, service)
            cache.sanity_check()

    def test_demand_stages_before_speculation_without_reordering_issued_reads(self):
        with hicache(self, namespace_keys=True) as owner:
            _, cache, service, _ = owner
            scopes = [cache_fixtures.target(range(i, i + 4)) for i in (1, 5, 9, 13, 17)]
            self._cold_scopes(owner, scopes)
            cc = cache.cache_controller
            reads = service.control.executor.storage_reads
            first, staged, joined, demand = [
                CacheRequestHandle(name, 0)
                for name in ("first", "staged", "joined", "demand")
            ]
            entered, release = threading.Event(), threading.Event()
            calls = []
            get = cc.page_get_func

            def observe_get(operation, *args, **kwargs):
                calls.append(operation.handle)
                if operation.handle == first:
                    entered.set()
                    assert release.wait(5)
                return get(operation, *args, **kwargs)

            def ready(condition):
                cache.drain_storage_control_queues()
                return condition()

            def parked(handle):
                info = cache.ongoing_prefetch.get(handle)
                return info is not None and getattr(
                    info.operation, "host_allocation_deferred", None
                )

            try:
                with (
                    patch.object(cc, "page_get_func", observe_get),
                    patch.object(cache, "prefetch_threshold", 1),
                    patch.object(cache, "prefetch_stop_policy", "wait_complete"),
                ):
                    self._ordinary_prefetch(cache, service, scopes[0], first)
                    wait_for(service, lambda: ready(entered.is_set))
                    self._ordinary_prefetch(cache, service, scopes[1], staged)
                    wait_for(
                        service, lambda: ready(lambda: cc.prefetch_buffer.qsize() == 1)
                    )
                    hint = self._restore(service, scopes[2], "speculation")
                    hint_handle = reads.interests[hint.ref.operation_id].job.handle
                    wait_for(service, lambda: ready(lambda: parked(hint_handle)))
                    shared = self._restore(service, scopes[3], "soon-demand")
                    shared_handle = reads.interests[shared.ref.operation_id].job.handle
                    wait_for(service, lambda: ready(lambda: parked(shared_handle)))
                    self._ordinary_prefetch(cache, service, scopes[3], joined)
                    self.assertEqual(reads.demands[joined].job.handle, shared_handle)
                    self._ordinary_prefetch(cache, service, scopes[4], demand)
                    wait_for(service, lambda: ready(lambda: parked(demand)))
                    release.set()
                    for request in (first, staged, joined, demand):
                        self.assertEqual(
                            self._consume_ordinary_prefetch(cache, service, request),
                            (4, 0),
                        )
                    self.assertEqual(wait(service, hint).status, Status.COMPLETED)
                    self.assertEqual(wait(service, shared).status, Status.COMPLETED)
            finally:
                release.set()
                for request in (first, staged, joined, demand):
                    cache.finish(request, CacheRequestOutcome.ABORT)
            assert_retired(cache, service)
            self.assertEqual(calls, [first, staged, shared_handle, demand, hint_handle])

    def test_joined_capacity_wait_defers_and_normal_prefetch_can_continue(self):
        with hicache(self, namespace_keys=True) as owner:
            _, cache, service, _ = owner
            target = cache_fixtures.target(cache_salt="pinned-fallback")
            self._cold_scopes(owner, [target])
            pressure = self._host_pressure(owner)
            self._pin_pressure(service, pressure)
            demand = CacheRequestHandle("capacity-fallback", 0)
            cc = cache.cache_controller
            queries = []
            query = cc._storage_hit_query

            def counted_query(operation):
                queries.append(operation)
                return query(operation)

            try:
                with (
                    patch.object(cc, "_storage_hit_query", counted_query),
                    patch.object(cache, "prefetch_threshold", 1),
                    patch.object(cache, "prefetch_stop_policy", "wait_complete"),
                ):
                    queued = self._restore(service, target, "pinned-native")
                    wait_for(
                        service,
                        lambda: (
                            current(service, queued).update.code
                            == "whole_scope_capacity"
                        ),
                    )
                    self._ordinary_prefetch(cache, service, target, demand)
                    self.assertIn(
                        demand, service.control.executor.storage_reads.demands
                    )
                    done = wait(service, queued)
                    self.assertEqual(done.status, Status.DEFERRED)
                    self.assertEqual(done.update.code, "whole_scope_capacity")
                    self.assertEqual(len(queries), 1)
                    self.assertEqual(
                        self._consume_ordinary_prefetch(cache, service, demand),
                        (0, None),
                    )
                    self.assertFalse(service.control.executor.storage_reads._joinable)
                    self.assertTrue(cache.check_prefetch_progress(demand))
                    self.assertFalse(cache.has_ongoing_prefetch(demand))
                    # A fresh ordinary attempt uses its own normal hit-allocation
                    # fallback; it cannot rejoin the retired native job.
                    self._ordinary_prefetch(cache, service, target, demand)
                    self.assertNotIn(
                        demand, service.control.executor.storage_reads.demands
                    )
                    self.assertEqual(
                        self._consume_ordinary_prefetch(cache, service, demand),
                        (0, None),
                    )
                    self.assertEqual(len(queries), 2)
                    self.assertTrue(cache.check_prefetch_progress(demand))
                    self.assertFalse(cache.has_ongoing_prefetch(demand))
                    self.assertEqual(cc.mem_pool_host.available_size(), 0)
            finally:
                cache.finish(demand, CacheRequestOutcome.ABORT)
                self._pin_pressure(service, pressure, mode="release")
            assert_retired(cache, service)
            cache.sanity_check()

    def test_ordinary_restore_survives_eviction_until_admission_or_abort(self):
        for finish in ("admission", "abort", "over_budget"):
            with (
                self.subTest(finish=finish),
                hicache(
                    self, namespace_keys=True, write_policy="write_through"
                ) as owner,
            ):
                _, cache, service, _ = owner
                target = cache_fixtures.target()
                self._cold_scopes(owner, [target])
                demand = CacheRequestHandle("handoff-demand", 0)
                cc = cache.cache_controller
                with (
                    patch.object(cache, "prefetch_threshold", 1),
                    patch.object(cache, "prefetch_stop_policy", "wait_complete"),
                    patch.object(
                        cc,
                        "prefetch_capacity_limit",
                        6 if finish == "over_budget" else 8,
                    ),
                ):
                    self._ordinary_prefetch(cache, service, target, demand)

                    def published():
                        cache.drain_storage_control_queues()
                        return demand not in cache.ongoing_prefetch

                    wait_for(service, published)
                    if finish == "over_budget":
                        self.assertEqual(cache.evict_host(4), 4)
                        cache.finish(demand, CacheRequestOutcome.ABORT)
                        continue
                    self.assertEqual(cache.evict_host(4), 0)
                    self.assertEqual(cache.pop_prefetch_loaded_span(demand), (4, 0))
                    self.assertEqual(cache.evict_host(4), 0)
                    if finish == "abort":
                        cache.finish(demand, CacheRequestOutcome.ABORT)
                    else:
                        cache.finish_storage_prefetch_admission(demand, 4, None)
                    self.assertEqual(cache.evict_host(4), 4)
                assert_retired(cache, service)
                cache.sanity_check()

    def test_ordinary_restore_retains_an_already_published_host_prefix(self):
        with hicache(self, namespace_keys=True, write_policy="write_through") as owner:
            _, cache, service, _ = owner
            target = cache_fixtures.target()
            self._cold_scopes(owner, [target])
            first = CacheRequestHandle("first-restore", 0)
            second = CacheRequestHandle("overlapping-restore", 0)
            reads = service.control.executor.storage_reads
            try:
                with (
                    patch.object(cache, "prefetch_threshold", 1),
                    patch.object(cache, "prefetch_stop_policy", "wait_complete"),
                    patch.object(cache.cache_controller, "prefetch_capacity_limit", 16),
                    # Exercise two already-issued reads, independent of coalescing.
                    patch.object(reads, "join_demand", return_value=False),
                ):
                    self._ordinary_prefetch(cache, service, target, first)
                    self._ordinary_prefetch(cache, service, target, second)

                    def published():
                        cache.drain_storage_control_queues()
                        return all(
                            req not in cache.ongoing_prefetch for req in (first, second)
                        )

                    wait_for(service, published)
                    self.assertEqual(cache.pop_prefetch_loaded_span(first), (4, 0))
                    self.assertEqual(cache.pop_prefetch_loaded_span(second), (0, None))
                    self.assertEqual(reads.snapshot()["admission_held_tokens"], 4)

                    cache.finish_storage_prefetch_admission(first, 4, None)
                    # The second request still needs the same resident pages.
                    self.assertEqual(cache.evict_host(4), 0)
                    cache.finish(second, CacheRequestOutcome.ABORT)
                    self.assertEqual(cache.evict_host(4), 4)
            finally:
                cache.finish(first, CacheRequestOutcome.ABORT)
                cache.finish(second, CacheRequestOutcome.ABORT)
            assert_retired(cache, service)
            cache.sanity_check()

    def test_ordinary_read_does_not_wait_for_a_rate_limited_hint(self):
        with hicache(self, namespace_keys=True) as owner:
            _, cache, service, _ = owner
            busy, target = [
                cache_fixtures.target(cache_salt=salt)
                for salt in ("rate-busy", "rate-queued")
            ]
            self._cold_scopes(owner, (busy, target))
            cc = cache.cache_controller
            entered, release = threading.Event(), threading.Event()
            get = cc.page_get_func
            reads = []
            demand = CacheRequestHandle("rate-limited-demand", 0)

            def blocked_get(operation, *args, **kwargs):
                reads.append(operation.request_id)
                entered.set()
                assert release.wait(5)
                return get(operation, *args, **kwargs)

            try:
                with (
                    patch.object(cc, "page_get_func", blocked_get),
                    patch.object(cc, "prefetch_capacity_limit", len(busy.token_ids)),
                    patch.object(cache, "prefetch_threshold", 1),
                    patch.object(cache, "prefetch_stop_policy", "wait_complete"),
                ):
                    active = self._restore(service, busy, "rate-active")
                    wait_for(service, entered.is_set)
                    queued = self._restore(service, target, "rate-queued")
                    self.assertEqual(queued.update.code, "storage_rate_limit")

                    self._ordinary_prefetch(cache, service, target, demand)
                    self.assertTrue(cache.check_prefetch_progress(demand))
                    self.assertFalse(cache.has_ongoing_prefetch(demand))
                    terminal = wait(service, queued)
                    self.assertEqual(terminal.status, Status.DEFERRED)
                    self.assertEqual(terminal.update.code, "ordinary_restore")

                    release.set()
                    self.assertEqual(wait(service, active).status, Status.COMPLETED)
                    self.assertEqual(len(reads), 1)
            finally:
                release.set()
                cache.finish(demand, CacheRequestOutcome.ABORT)
            assert_retired(cache, service)
            cache.sanity_check()

    def test_resident_system_prompt_does_not_serialize_cold_siblings(self):
        with hicache(self, namespace_keys=True) as owner:
            _, cache, service, _ = owner
            left, right = [
                cache_fixtures.target(tokens) for tokens in ((1, 2, 3), (1, 4, 5))
            ]
            resident = cache_fixtures.target((1, 8, 9))
            self._cold_scopes(owner, (left, right, resident))
            prefix = self._restore(service, resident, "prompt")
            self.assertEqual(wait(service, prefix).status, Status.COMPLETED)
            a = self._restore(service, left, "left")
            b = self._restore(service, right, "right")
            self.assertEqual([a.status, b.status], [Status.RUNNING, Status.RUNNING])
            self.assertEqual(
                [op.operation.storage_start for op in cache.ongoing_prefetch.values()],
                [1, 1],
            )
            self.assertEqual(wait(service, a).status, Status.COMPLETED)
            self.assertEqual(wait(service, b).status, Status.COMPLETED)
            assert_retired(cache, service)

    def test_deadline_keeps_pipeline_turn_until_query_or_get_actually_drains(self):
        for phase in ("query", "get"):
            config = ServiceConfig(
                namespace="ns",
                engine_id="engine",
                cache_limits=CacheLimits(max_concurrent_storage_reads=1),
            )
            with (
                self.subTest(phase=phase),
                hicache(self, config, namespace_keys=True) as owner,
            ):
                _, cache, service, _ = owner
                scopes = [
                    cache_fixtures.target(),
                    cache_fixtures.target(cache_salt="after-deadline"),
                ]
                self._cold_scopes(owner, scopes)
                cc = cache.cache_controller
                entered, release = threading.Event(), threading.Event()
                query, get = cc._storage_hit_query, cc.page_get_func
                first_operation, queries = [], []

                def blocked_query(operation):
                    queries.append(operation)
                    if not first_operation:
                        first_operation.append(operation)
                        if phase == "query":
                            entered.set()
                            assert release.wait(5)
                    return query(operation)

                def blocked_get(operation, *args, **kwargs):
                    if operation is first_operation[0] and phase == "get":
                        entered.set()
                        assert release.wait(5)
                    return get(operation, *args, **kwargs)

                try:
                    with (
                        patch.object(cc, "_storage_hit_query", blocked_query),
                        patch.object(cc, "page_get_func", blocked_get),
                    ):
                        first = public_transfer(
                            service, "prefetch", "storage", "host", "deadline-first"
                        )
                        wait_for(service, entered.is_set)
                        second = self._restore(service, scopes[1], "after-deadline")
                        self.clock.now += 5.1
                        service.progress()
                        self.assertEqual(
                            current(service, first).status, Status.CANCELLING
                        )
                        for _ in range(5):
                            service.progress()
                        reads = service.control.executor.storage_reads
                        self.assertIsNone(
                            reads.interests[second.ref.operation_id].job.operation
                        )
                        self.assertFalse(first_operation[0].native_quiescent)
                        self.assertEqual(len(queries), 1)
                        release.set()
                        self.assertEqual(wait(service, first).status, Status.CANCELLED)
                        self.assertEqual(wait(service, second).status, Status.COMPLETED)
                        self.assertEqual(len(queries), 2)
                finally:
                    release.set()
                assert_retired(cache, service)
                cache.sanity_check()

    def test_query_anchor_loss_defers_without_reading_into_stale_rows(self):
        with hicache(self, namespace_keys=True) as owner:
            cache, service, _ = cold_prefix(owner)
            prefix = transfer(
                service, src="storage", dst="host", tokens=(1, 2), message="prefix"
            )
            self.assertEqual(wait(service, prefix).status, Status.COMPLETED)
            cc = cache.cache_controller
            query = cc._storage_hit_query
            entered, release = threading.Event(), threading.Event()

            def blocked_query(operation):
                entered.set()
                assert release.wait(5)
                return query(operation)

            try:
                with (
                    patch.object(cc, "_storage_hit_query", blocked_query),
                    patch.object(cc, "page_get_func", wraps=cc.page_get_func) as get,
                ):
                    result = transfer(
                        service, src="storage", dst="host", message="lost-anchor"
                    )
                    wait_for(service, entered.is_set)
                    self.assertEqual(cache.evict_host(2), 2)
                    release.set()
                    done = wait(service, result)
                    self.assertEqual(done.status, Status.DEFERRED)
                    self.assertEqual(done.update.code, "source_changed")
                    get.assert_not_called()
            finally:
                release.set()

            assert_retired(cache, service)
            cache.sanity_check()

    def _host_pressure(self, owner, *, leave_free=0):
        """Fill real host rows via native HiCache backup; keep GPU empty."""
        _, cache, service, dma = owner
        pool = cache.cache_controller.mem_pool_host
        nodes = []
        while pool.available_size() > leave_free:
            count = min(
                2 if not nodes else cache.token_to_kv_pool_allocator.size,
                pool.available_size() - leave_free,
            )
            tokens = tuple(
                range(1000 + len(nodes) * 100, 1000 + len(nodes) * 100 + count)
            )
            node = seed(cache, tokens, value=11 + len(nodes))
            backup_prefix(cache, dma, node, evict=True)
            self.assertIsNone(node.component_data[FULL].value)
            self.assertIsNotNone(node.component_data[FULL].host_value)
            nodes.append((node, tokens))
        self.assertEqual(pool.available_size(), leave_free)
        return nodes

    def _pin_pressure(self, service, rows, *, mode="acquire", salt=None):
        for node, tokens in rows:
            payload = {
                "target": msgspec.json.decode(
                    msgspec.json.encode(cache_fixtures.target(tokens, cache_salt=salt))
                ),
                "location": {"tier": "host"},
                "mode": mode,
                "claim_id": f"pressure-{node.id}",
                "revision": 1 if mode == "acquire" else 2,
            }
            if mode == "acquire":
                payload["ttl_seconds"] = 10
            result = self.submit(service.control, "pin", payload, caller="keeper")
            self.assertEqual(result.status, Status.COMPLETED)

    def test_storage_prefetch_reclaims_only_shortfall_and_holds_existing_prefix(self):
        """A full host cache must not strand a valid cold prefix behind admission."""
        for existing in (0, 2):
            with (
                self.subTest(existing=existing),
                hicache(self, namespace_keys=True) as owner,
            ):
                cache, service, _ = cold_prefix(owner)
                prefix = None
                if existing:
                    restored = transfer(
                        service,
                        src="storage",
                        dst="host",
                        tokens=(1, 2),
                        message="prefix",
                    )
                    self.assertEqual(wait(service, restored).status, Status.COMPLETED)
                    prefix = prefix_path(cache, service)[0][0]
                pool = cache.cache_controller.mem_pool_host
                rows = self._host_pressure(owner, leave_free=2 - existing)
                first = rows[0][0]
                source_indices = (
                    prefix.component_data[FULL].host_value.clone()
                    if prefix is not None
                    else None
                )
                gpu_free = cache.token_to_kv_pool_allocator.available_size()
                with patch.object(
                    cache, "evict_host", wraps=cache.evict_host
                ) as evictions:
                    result = transfer(
                        service, src="storage", dst="host", message="restore"
                    )
                    self.assertIsNotNone(
                        next(
                            iter(service.control.executor.storage_reads.jobs.values())
                        ).operation
                    )
                    evictions.assert_not_called()
                    done = wait(service, result)
                    self.assertEqual(evictions.call_args_list[0].args, (2,))
                self.assertEqual(done.status, Status.COMPLETED)
                self.assertEqual(evictions.call_count, 1)
                self.assertIsNone(first.component_data[FULL].host_value)
                self.assertTrue(
                    all(
                        node.component_data[FULL].host_value is not None
                        for node, _ in rows[1:]
                    )
                )
                self.assertEqual(pool.available_size(), 0)
                self.assertEqual(
                    cache.token_to_kv_pool_allocator.available_size(), gpu_free
                )
                if prefix is not None:
                    self.assertTrue(
                        torch.equal(
                            prefix.component_data[FULL].host_value, source_indices
                        )
                    )
                host = cache.host_pool_group.anchor_entry.host_pool
                for node, _, _, _ in prefix_path(cache, service):
                    cd = node.component_data[FULL]
                    self.assertTrue(torch.all(host.k_buffer[0][cd.host_value] == 7))
                    self.assertEqual(cd.host_lock_ref, 0)
                    self.assertEqual(cd.lock_ref, 0)
                assert_retired(cache, service)
                cache.sanity_check()

    def test_pinned_capacity_waits_without_idle_rescans_and_live_sibling_outlives_first_timeout(
        self,
    ):
        with hicache(self, namespace_keys=True) as owner:
            cache, service, _ = cold_prefix(owner)
            rows = self._host_pressure(owner)
            self._pin_pressure(service, rows)
            held = {
                node.id: node.component_data[FULL].host_value.clone()
                for node, _ in rows
            }
            cc = cache.cache_controller
            with (
                patch.object(cache, "evict_host", wraps=cache.evict_host) as evictions,
                patch.object(
                    cc, "_storage_hit_query", wraps=cc._storage_hit_query
                ) as queries,
            ):
                first = public_transfer(
                    service, "prefetch", "storage", "host", "first", caller="A"
                )
                second = transfer(
                    service, src="storage", dst="host", caller="B", message="second"
                )
                other = transfer(
                    service,
                    src="storage",
                    dst="host",
                    caller="C",
                    message="other",
                    tokens=(1, 2),
                )
                wait_for(
                    service,
                    lambda: (
                        current(service, first).update.code == "whole_scope_capacity"
                    ),
                )
                for _ in range(20):
                    service.progress()
                self.assertEqual(evictions.call_count, 1)
                self.assertEqual(queries.call_count, 1)

                self.assertEqual(
                    current(service, first).update.code, "whole_scope_capacity"
                )
                self.assertEqual(
                    cache.cache_controller.mem_pool_host.available_size(), 0
                )
                self.assertTrue(
                    all(
                        torch.equal(node.component_data[FULL].host_value, held[node.id])
                        for node, _ in rows
                    )
                )
                self.assertEqual(
                    service.control.cancel(other.ref).status, Status.CANCELLED
                )
                # Expire only A's operation. B still owns the shared physical
                # preparation; native PIN deadlines remain at t=110.
                self._pin_pressure(service, rows[1:2], mode="release")
                self.clock.now = 105.1
                service.progress()
                self.assertEqual(current(service, first).status, Status.CANCELLED)
                self.assertEqual(wait(service, second).status, Status.COMPLETED)
                self.assertEqual(evictions.call_count, 2)
                self.assertEqual(queries.call_count, 1)
                self.assertTrue(
                    all(
                        torch.equal(node.component_data[FULL].host_value, held[node.id])
                        for node, _ in rows[:1] + rows[2:]
                    )
                )
            self._pin_pressure(service, rows[:1] + rows[2:], mode="release")
            assert_retired(cache, service)
            cache.sanity_check()

    def test_request_with_backend_actions_keeps_its_own_storage_read(self):
        hint = KvHintsEnvelope(
            protocol_version="0.1",
            message_id="backend-request",
            actions=[
                KvHintAction(
                    action_id="a", action_type="backend.test", action_version="1.0"
                )
            ],
        )
        with hicache(self, namespace_keys=True) as owner:
            cache, service, _ = cold_prefix(owner)
            request = CacheRequestHandle("backend-demand", 0)
            try:
                with (
                    patch.object(
                        cache.cache_controller.storage_backend,
                        "consumes_kv_hints",
                        return_value=True,
                    ),
                    patch.object(
                        service.control.executor.storage_reads,
                        "join_demand",
                        return_value=True,
                    ) as join,
                    patch.object(cache, "prefetch_stop_policy", "wait_complete"),
                    patch.object(cache, "prefetch_threshold", 1),
                ):
                    cache.prefetch_from_storage(
                        request,
                        cache.tree_core.root_node.id,
                        cache_fixtures.array("q", [1, 2, 3, 4]),
                        matched_prefix_tokens=[],
                        cache_salt=service.cache_salt(None),
                        kv_hints=hint,
                    )
                    join.assert_not_called()
                    op = cache.ongoing_prefetch[request].operation
                    self.assertEqual(op.storage_read_metadata["kv_hints"], hint)
                    native = self._restore(
                        service, cache_fixtures.target(), "late-hint"
                    )
                    reads = service.control.executor.storage_reads
                    job = reads.interests[native.ref.operation_id].job
                    self.assertIsNot(job.operation, op)
                    self.assertEqual(wait(service, native).status, Status.COMPLETED)
                    self.assertEqual(
                        self._consume_ordinary_prefetch(cache, service, request), (4, 0)
                    )
            finally:
                cache.finish(request, CacheRequestOutcome.ABORT)
            assert_retired(cache, service)

    def test_demand_and_hint_share_backend_io_in_either_order(self):
        for hint_first in (True, False):
            with (
                self.subTest(hint_first=hint_first),
                hicache(self, namespace_keys=True, write_policy="write_through") as owner,
            ):
                cache, service, _ = cold_prefix(owner)
                controller = cache.cache_controller
                entered, release = threading.Event(), threading.Event()
                demand = CacheRequestHandle("demand", 0)
                target = cache_fixtures.target()
                try:
                    with (
                        patch.object(
                            controller,
                            "_storage_hit_query",
                            wraps=controller._storage_hit_query,
                        ) as query,
                        patch.object(
                            controller,
                            "page_get_func",
                            side_effect=block_io(
                                controller.page_get_func, entered, release
                            ),
                        ) as get,
                        patch.object(cache, "prefetch_threshold", 1),
                        patch.object(cache, "prefetch_stop_policy", "wait_complete"),
                    ):
                        if hint_first:
                            hint = self._restore(service, target, "hint")
                        else:
                            self._ordinary_prefetch(cache, service, target, demand)

                        def started():
                            cache.drain_storage_control_queues()
                            return entered.is_set()

                        wait_for(service, started)
                        if hint_first:
                            self._ordinary_prefetch(cache, service, target, demand)
                        else:
                            hint = self._restore(service, target, "hint")
                        reads = service.control.executor.storage_reads
                        job = reads.interests[hint.ref.operation_id].job
                        self.assertIs(
                            job,
                            reads.demands[demand].job
                            if hint_first
                            else reads.jobs[demand],
                        )
                        release.set()
                        settled = wait(service, hint)
                        self.assertEqual(settled.status, Status.COMPLETED)
                        self.assertEqual(
                            settled.public_result.ready_scope[-1].coverage.end_token, 4
                        )
                        self.assertEqual(
                            self._consume_ordinary_prefetch(cache, service, demand),
                            (4, 0),
                        )
                        self.assertEqual((query.call_count, get.call_count), (1, 1))
                        self.assertEqual(cache.evict_host(4), 0)
                        cache.finish_storage_prefetch_admission(demand, 4, None)
                        self.assertEqual(cache.evict_host(4), 4)
                finally:
                    release.set()
                    cache.finish(demand, CacheRequestOutcome.ABORT)
                assert_retired(cache, service)

    def test_fragmented_long_demand_resumes_after_control_walk_budget_exhaustion(self):
        from sglang.srt.managers.kv_control.contracts import ControlError
        from sglang.srt.mem_cache.base_prefix_cache import (
            MatchPrefixParams,
        )

        config = ServiceConfig(
            namespace="ns",
            limits=Limits(max_scope_tokens=1),
            cache_limits=CacheLimits(max_query_nodes=1),
        )
        with hicache(self, config, namespace_keys=True) as owner:
            _, cache, service, dma = owner
            self._cold_scopes(owner, [cache_fixtures.target((1, 2, 3, 4, 5, 6))])
            request = CacheRequestHandle("long-demand", 0)
            tokens = cache_fixtures.array("q", range(1, 65538))
            try:
                with (
                    patch.object(cache, "prefetch_threshold", 1),
                    patch.object(cache, "prefetch_stop_policy", "wait_complete"),
                ):
                    hint = self._restore(
                        service, cache_fixtures.target((1,)), "ancestor"
                    )
                    salt = service.cache_salt(None)
                    cache.prefetch_from_storage(
                        request,
                        cache.tree_core.root_node.id,
                        tokens,
                        matched_prefix_tokens=[],
                        cache_salt=salt,
                    )
                    executor = service.control.executor
                    demand = executor.storage_reads.demands[request]
                    self.assertEqual(wait(service, hint).status, Status.COMPLETED)
                    # An independent host publication extends the shared ancestor.
                    backup_prefix(cache, dma, seed(cache, (1, 2)), evict=True)
                    with self.assertRaises(ControlError) as raised:
                        executor._walk(tokens, demand.job.compatibility)
                    self.assertEqual(raised.exception.code, "capacity")
                    self.assertFalse(cache.check_prefetch_progress(request))
                    self.assertEqual(
                        cache.ongoing_prefetch[request].operation.storage_start, 2
                    )
                    self.assertEqual(
                        self._consume_ordinary_prefetch(cache, service, request), (1, 0)
                    )
                    match = cache.match_prefix(
                        MatchPrefixParams(
                            key=cache_fixtures.RadixKey(tokens, cache_salt=salt)
                        )
                    )
                    self.assertEqual(match.full_kv_hit_length, 6)
            finally:
                cache.finish(request, CacheRequestOutcome.ABORT)
            assert_retired(cache, service)

    def test_native_demand_partial_failure_and_abort_leave_no_holds(self):
        from sglang.srt.managers.schedule_batch import split_cached_prefix_by_tier

        with hicache(self, namespace_keys=True) as owner:
            cache, service, dma = cold_prefix(owner)
            cc = cache.cache_controller
            entered, release = threading.Event(), threading.Event()
            original = cc.page_get_func
            calls = []

            def partial(operation, hashes, indices, extra_info=None):
                calls.append(1)
                entered.set()
                assert release.wait(5)
                # A real complete leading page, followed by a failed page read.
                return original(operation, hashes[:2], indices[:2], extra_info)

            demand = CacheRequestHandle("partial", 0)
            try:
                with (
                    patch.object(cc, "page_get_func", partial),
                    patch.object(cache, "prefetch_threshold", 1),
                    patch.object(cache, "prefetch_stop_policy", "wait_complete"),
                ):
                    native = transfer(
                        service, src="storage", dst="host", message="partial"
                    )
                    wait_for(service, entered.is_set)
                    cache.prefetch_from_storage(
                        demand,
                        cache.tree_core.root_node.id,
                        cache_fixtures.array("q", [1, 2, 3, 4]),
                        matched_prefix_tokens=[],
                        cache_salt=service.cache_salt(None),
                    )
                    release.set()
                    done = wait(service, native)
                    self.assertEqual(done.status, Status.FAILED)
                    self.assertTrue(cache.check_prefetch_progress(demand))
                    span = cache.pop_prefetch_loaded_span(demand)
                    self.assertEqual(span, (2, 0))
                    self.assertEqual(
                        split_cached_prefix_by_tier(2, 2, *span), (0, 0, 2)
                    )
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(cache.evict_host(4), 2)
            finally:
                release.set()
                cache.finish(demand, CacheRequestOutcome.ABORT)
            assert_retired(cache, service)


if __name__ == "__main__":
    unittest.main()

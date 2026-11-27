# SPDX-License-Identifier: Apache-2.0
"""Host DEMOTE uses actual file I/O, source receipts and CPU pool reclamation."""

import threading
import unittest
from unittest.mock import patch

import test_kv_control_cache as cache_fixtures
import torch
from kv_control_test_utils import (
    FULL,
    backup_prefix,
    block_io,
    current,
    hicache,
    mooncake_store,
    public_transfer,
    seed,
    start_inference_load,
    wait,
)
from sglang.srt.managers.kv_control.contracts import Status
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def backed_up_host(owner):
    _, cache, service, dma = owner
    node = seed(cache)
    backup_prefix(cache, dma, node, evict=True)
    pool = cache.host_pool_group.anchor_entry.host_pool
    return cache, service, dma, node, pool, pool.available_size()


def ranges(outcomes):
    return [(o.coverage.start_token, o.coverage.end_token) for o in outcomes]


class TestKVControlHostDemote(cache_fixtures.KVCacheTestCase):
    def test_legacy_file_positive_metadata_does_not_advertise_host_reclamation(self):
        from sglang.srt.mem_cache.hicache_storage import MetadataCache

        with hicache(self, storage=True) as owner:
            _, cache, service, _ = owner
            backend = cache.cache_controller.storage_backend
            self.assertIsNone(backend._unified_suffixes)
            with patch.object(backend, "metadata_cache", MetadataCache(60)):
                caps = service.capabilities()["public_protocol"]
                self.assertNotIn(("host", "storage"), caps["paths"]["sglang.demote"])
                result = public_transfer(service, "demote", "host", "storage", "cold")
                self.assertEqual(result.status, Status.UNSUPPORTED)

    def test_reclaims_full_host_scope_and_recovers_file_bytes_in_both_write_policies(
        self,
    ):
        for policy in ("write_back", "write_through"):
            with (
                self.subTest(policy=policy),
                hicache(self, storage=True, write_policy=policy) as owner,
            ):
                _, cache, service, dma = owner
                node = seed(cache)
                backup_prefix(cache, dma, node, evict=True)
                pool = cache.host_pool_group.anchor_entry.host_pool
                initial = pool.available_size()
                self.assertIn(
                    ("host", "storage"),
                    service.capabilities()["public_protocol"]["paths"]["sglang.demote"],
                )
                result = public_transfer(service, "demote", "host", "storage", "cold")
                done = wait(service, result)
                self.assertEqual(done.status, Status.COMPLETED, done)
                self.assertEqual(ranges(done.public_result.preserved_scope), [(0, 4)])
                self.assertEqual(ranges(done.public_result.reclaimed_scope), [(0, 4)])
                self.assertEqual(pool.available_size(), initial + 4)
                self.assertFalse(cache.tree_core.root_node.children)
                self.assertNotIn(node, cache.tree_core.evictable_host_leaves)
                cache.sanity_check()
                # Recovery must publish the exact persisted K/V, then survive
                # H2D back to a request-usable device allocation.
                loaded = public_transfer(service, "prefetch", "storage", "host", "read")
                self.assertEqual(wait(service, loaded).status, Status.COMPLETED)
                rows = service.control.executor._walk(
                    (1, 2, 3, 4),
                    service.control.executor._compatibility(cache_fixtures.target()),
                )
                start_inference_load(cache, rows[-1][0])
                dma.finish()
                cache.loading_check()
                indices = torch.cat([n.component_data[FULL].value for n, *_ in rows])
                device = cache.token_to_kv_pool_allocator.get_kvcache()
                self.assertTrue(torch.all(device.k_buffer[0][indices] == 7))
                self.assertTrue(torch.all(device.v_buffer[0][indices] == 8))
                self.assertEqual(current(service, result), done)
                cache.sanity_check()

    def test_shared_ancestor_is_preserved_and_only_selected_suffix_reclaimed(self):
        with hicache(self, storage=True, write_policy="write_through") as owner:
            _, cache, service, dma = owner
            ancestor = seed(cache, (1, 2))
            leaf = seed(cache)
            sibling = seed(cache, (1, 2, 5, 6), value=11)
            backup_prefix(cache, dma, leaf, evict=True)
            backup_prefix(cache, dma, sibling, evict=True)
            pool = cache.host_pool_group.anchor_entry.host_pool
            initial = pool.available_size()
            with patch.object(cache.cache_controller, "write_storage") as write:
                blocked = public_transfer(
                    service, "demote", "host", "storage", "ancestor", tokens=(1, 2)
                )
                self.assertEqual(blocked.status, Status.DEFERRED)
                self.assertEqual(blocked.update.code, "structural_dependency")
                write.assert_not_called()
            result = public_transfer(service, "demote", "host", "storage", "cold")
            done = wait(service, result)
            self.assertEqual(done.status, Status.PARTIAL)
            self.assertEqual(
                ranges(done.public_result.preserved_scope), [(0, 2), (2, 4)]
            )
            self.assertEqual(ranges(done.public_result.reclaimed_scope), [(2, 4)])
            self.assertEqual(
                [
                    (o.coverage.end_token, o.code)
                    for o in done.public_result.blocked_scope
                ],
                [(2, "structural_dependency")],
            )
            self.assertEqual(pool.available_size(), initial + 2)
            self.assertIsNotNone(ancestor.component_data[FULL].host_value)
            self.assertIsNotNone(sibling.component_data[FULL].host_value)
            self.assertIsNone(leaf.component_data[FULL].host_value)
            cache.sanity_check()

    def test_radix_split_during_write_retains_ack_geometry_and_reclaims_each_slot_once(
        self,
    ):
        from array import array

        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import RadixKey

        with hicache(self, storage=True) as owner:
            cache, service, _, _, pool, initial = backed_up_host(owner)
            entered, release = threading.Event(), threading.Event()
            cc = cache.cache_controller
            blocked = block_io(cc.page_set_func, entered, release)

            try:
                with patch.object(cc, "page_set_func", blocked):
                    result = public_transfer(
                        service, "demote", "host", "storage", "cold"
                    )
                    self.assertTrue(entered.wait(5))
                    cache.match_prefix(
                        MatchPrefixParams(key=RadixKey(array("q", [1, 2])))
                    )
                    self.assertEqual(len(cache.tree_core.root_node.children), 1)
                    release.set()
                    done = wait(service, result)
            finally:
                release.set()
            self.assertEqual(done.status, Status.COMPLETED)
            self.assertEqual(ranges(done.public_result.reclaimed_scope), [(0, 4)])
            self.assertEqual(pool.available_size(), initial + 4)
            self.assertFalse(cache.tree_core.root_node.children)
            # Exactly replaying an operation uses its immutable receipt.
            replayed = public_transfer(service, "demote", "host", "storage", "cold")
            self.assertEqual(replayed, done)
            self.assertEqual(pool.available_size(), initial + 4)
            cache.sanity_check()

    def test_foreign_h2d_keeps_host_source_until_its_own_fence(self):
        with hicache(self, storage=True, write_policy="write_through") as owner:
            _, cache, service, dma = owner
            node = seed(cache)
            backup_prefix(cache, dma, node, evict=True)
            pool = cache.host_pool_group.anchor_entry.host_pool
            initial = pool.available_size()
            entered, release = threading.Event(), threading.Event()
            cc = cache.cache_controller
            blocked = block_io(cc.page_set_func, entered, release)

            try:
                with patch.object(cc, "page_set_func", blocked):
                    result = public_transfer(
                        service, "demote", "host", "storage", "cold"
                    )
                    self.assertTrue(entered.wait(5))
                    start_inference_load(cache, node)
                    self.assertTrue(dma.pending)
                    release.set()
                    done = wait(service, result)
            finally:
                release.set()
            self.assertEqual(done.status, Status.PARTIAL)
            self.assertEqual(
                {o.code for o in done.public_result.blocked_scope}, {"inflight_load"}
            )
            self.assertEqual(pool.available_size(), initial)
            self.assertIsNotNone(node.component_data[FULL].host_value)
            dma.finish()
            cache.loading_check()
            self.assertTrue(
                torch.all(
                    cache.token_to_kv_pool_allocator.get_kvcache().k_buffer[0][
                        node.component_data[FULL].value
                    ]
                    == 7
                )
            )
            cache.sanity_check()

    def test_partial_write_keeps_all_source_slots_and_reports_actual_saved_prefix(self):
        with hicache(self, storage=True) as owner:
            cache, service, _, node, pool, initial = backed_up_host(owner)
            cc = cache.cache_controller
            original, calls = cc.page_set_func, 0

            def partial(*args, **kwargs):
                nonlocal calls
                calls += 1
                return original(*args, **kwargs) if calls == 1 else False

            with (
                patch(
                    "sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller.STORAGE_BATCH_SIZE",
                    2,
                ),
                patch.object(cc, "page_set_func", partial),
            ):
                done = wait(
                    service,
                    public_transfer(service, "demote", "host", "storage", "partial"),
                )
            self.assertEqual(done.status, Status.FAILED)
            self.assertEqual(done.update.code, "incomplete_storage_write")
            self.assertEqual(ranges(done.public_result.preserved_scope), [(0, 2)])
            self.assertEqual(ranges(done.public_result.missing_scope), [(2, 4)])
            self.assertFalse(done.public_result.reclaimed_scope)
            self.assertEqual(pool.available_size(), initial)
            self.assertIsNotNone(node.component_data[FULL].host_value)
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            cache.sanity_check()

    def test_short_preexisting_mooncake_object_blocks_host_reclamation(self):
        from types import SimpleNamespace

        with hicache(self, storage=True) as owner:
            _, cache, service, dma = owner
            node = seed(cache)
            backup_prefix(cache, dma, node, evict=True)
            cc = cache.cache_controller
            store = mooncake_store(cache)
            pool = store.mem_pool_host
            initial = pool.available_size()
            hashes = cache.tree_core.get_hash_values(node.id)
            objects, _, sizes = store._batch_preprocess(
                store._tag_keys(hashes), node.component_data[FULL].host_value
            )
            complete_sizes = dict(zip(objects, sizes, strict=True))
            complete_sizes[objects[-1]] -= 1
            store.store = SimpleNamespace(
                batch_is_exist=lambda keys: [int(k in complete_sizes) for k in keys],
                get_size=complete_sizes.__getitem__,
            )
            store.config = SimpleNamespace(
                master_server_address="test", metadata_server="test", tenant_id="test"
            )
            with (
                patch.object(cc, "storage_backend", store),
                patch.object(cc, "storage_backend_type", "mooncake"),
                patch.object(cc, "page_set_func", cc._page_set_zero_copy),
                patch.object(pool, "layout", "page_first_direct"),
            ):
                result = public_transfer(
                    service, "demote", "host", "storage", "truncated"
                )
                done = wait(service, result)
            self.assertEqual(done.update.code, "incomplete_storage_write")
            self.assertEqual(ranges(done.public_result.preserved_scope), [(0, 3)])
            self.assertFalse(done.public_result.reclaimed_scope)
            self.assertEqual(pool.available_size(), initial)
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            cache.sanity_check()

    def test_storage_clear_after_worker_verification_fences_host_reclamation(self):
        with hicache(self, storage=True) as owner:
            cache, service, _, node, pool, initial = backed_up_host(owner)
            result = public_transfer(service, "demote", "host", "storage", "clear")
            # The worker has certified the objects. Hold its ACK off the owner
            # queue until the clear has invalidated that verification.
            queue = cache.cache_controller.ack_backup_queue
            ack = queue.get(timeout=5)
            self.assertEqual(ack.completed_tokens, 4)
            try:
                self.assertTrue(cache.clear_storage_backend())
            finally:
                queue.put(ack)
            done = wait(service, result)
            self.assertEqual(done.update.code, "storage_cleared")
            self.assertEqual(ranges(done.public_result.blocked_scope), [(0, 4)])
            self.assertFalse(done.public_result.reclaimed_scope)
            self.assertFalse(done.public_result.preserved_scope)
            self.assertIsNotNone(node.component_data[FULL].host_value)
            self.assertEqual(pool.available_size(), initial)
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            cache.sanity_check()

    def test_cancel_waits_for_storage_quiescence_and_never_reclaims_source(self):
        with hicache(self, storage=True) as owner:
            cache, service, _, node, pool, initial = backed_up_host(owner)
            entered, release = threading.Event(), threading.Event()
            cc = cache.cache_controller
            original = cc.page_set_func

            def blocked(*args, **kwargs):
                success = original(*args, **kwargs)
                entered.set()
                assert release.wait(5)
                return success

            try:
                with patch.object(cc, "page_set_func", blocked):
                    result = public_transfer(
                        service, "demote", "host", "storage", "cold"
                    )
                    self.assertTrue(entered.wait(5))
                    cancelled = service.control.cancel(result.ref)
                    self.assertFalse(cancelled.update.quiescent)
                    self.assertGreater(node.component_data[FULL].host_lock_ref, 0)
                    self.assertEqual(pool.available_size(), initial)
                    release.set()
                    done = wait(service, result)
            finally:
                release.set()
            self.assertEqual(done.status, Status.CANCELLED)
            self.assertTrue(done.public_result.preserved_scope)
            self.assertFalse(done.public_result.reclaimed_scope)
            self.assertEqual(pool.available_size(), initial)
            self.assertEqual(node.component_data[FULL].host_lock_ref, 0)
            self.assertEqual(node.component_data[FULL].lock_ref, 0)
            cache.sanity_check()


if __name__ == "__main__":
    unittest.main()

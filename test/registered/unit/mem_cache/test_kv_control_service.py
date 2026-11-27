# SPDX-License-Identifier: Apache-2.0
"""Serving control, request completion and static model namespaces."""

import unittest
from array import array
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import msgspec
import pytest
import test_kv_control_cache as cache_fixtures

from sglang.srt.managers.io_struct import (
    CheckWeightsReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightVersionReqInput,
)
from sglang.srt.managers.kv_control.contracts import TokensTarget, resolve_prefix
from sglang.srt.managers.kv_control.key_namespace import encode_cache_salt
from sglang.srt.managers.kv_control.service import (
    KVControlService,
    ServiceConfig,
    dispatch,
    runtime_config,
)
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def serving_request(rid="same", *, salt=None):
    return Req(
        rid=rid,
        origin_input_text=None,
        origin_input_ids=array("q", [1, 2, 3, 4]),
        sampling_params=SamplingParams(max_new_tokens=1),
        vocab_size=128,
        cache_salt=salt,
    )


@contextmanager
def service_owner(case, *, namespace_keys=False):
    cache, _ = case.make(session=False)
    cache.reset()
    service = KVControlService(
        cache,
        ServiceConfig(namespace="ns", engine_id="engine"),
        clock=case.clock,
        namespace_keys=namespace_keys,
    )
    yield (cache, service)
    assert not service.has_pending


insert, match_len, target = (
    cache_fixtures.insert,
    cache_fixtures.match_len,
    cache_fixtures.target,
)


class TestKVControlService(cache_fixtures.KVCacheTestCase):
    def test_owner_watermark_orders_real_ready_remove_and_scoped_reconciliation(self):
        import test_kv_control_transfers as transfer_fixtures
        from sglang.srt.managers.kv_control.contracts import Status

        with transfer_fixtures.hicache(self, storage=True) as owner:
            _, cache, service, dma = owner
            self.assertFalse(cache.tree_core.kv_events.enabled)
            before = service.cache_watermark()
            node = transfer_fixtures.seed(cache)
            self.assertGreater(service.cache_watermark().version, before.version)
            transfer_fixtures.persist_prefix(cache, service, dma, node)
            self.assertEqual(cache.evict_host(4), 4)
            result = transfer_fixtures.transfer(service)
            complete = transfer_fixtures.wait(service, result)
            node = transfer_fixtures.prefix_path(cache, service)[-1][0]
            self.assertEqual(complete.status, Status.COMPLETED)
            ready = complete.cache_watermark
            self.assertEqual(ready, service.cache_watermark())
            self.assertEqual(cache.evict_host(100), 4)
            self.assertIsNone(
                node.component_data[cache_fixtures.ComponentType.FULL].host_value
            )
            self.assertGreater(service.cache_watermark().version, ready.version)
            scoped = dispatch(
                service,
                {
                    "kind": "residency",
                    "context": msgspec.to_builtins(service.control.context("A")),
                    "target": msgspec.json.decode(
                        msgspec.json.encode(cache_fixtures.target())
                    ),
                    "location": {"tier": "host"},
                },
            )["data"]
            self.assertFalse(scoped["items"])
            self.assertTrue(scoped["missing"])
            self.assertEqual(scoped["coverage_semantics"], "requested_scope_only")
            self.assertGreater(scoped["cache_watermark"]["version"], ready.version)
            # The historical result is unchanged even when its replica is gone.
            self.assertEqual(service.control.registry.get(result.ref), complete)
            self.assertEqual(cache.take_events(), [])
            cache.sanity_check()

    def test_cache_reset_advances_generation_and_fences_old_control_context(self):
        with service_owner(self) as (cache, service):
            before = service.cache_watermark()
            context = service.control.context("router")
            cache.reset()
            service.reset_after_cache_flush()
            current = service.capabilities()["cache_watermark"]
            self.assertGreater(current.generation, before.generation)
            self.assertEqual(current.source_epoch, before.source_epoch)
            reply = dispatch(service, {"kind": "capacity", "context": context})
            self.assertEqual(reply["error"]["code"], "stale_epoch")

    def test_request_cached_bounds_committed_prefix_to_whole_pages(self):
        from sglang.srt.managers.kv_control.contracts import MAX_SCOPE_TOKENS

        cache, _ = self.make(session=False, page_size=64)
        cache.reset()
        service = KVControlService(
            cache, ServiceConfig(namespace="ns"), clock=self.clock
        )
        for n, expected in ((63, 0), (65, 64), (70000, 65536)):
            req = serving_request()
            req.full_untruncated_fill_ids = array("q", range(n))
            req.finished_reason = FINISH_LENGTH(1)
            req.kv_control_context = service.control.context(
                "request", binding_id=str(n)
            )
            with patch.object(service.control, "request_event") as finish:
                service.finish_request(req, committed_len=n, checkpointed=True)
            committed = finish.call_args.kwargs["committed"]
            if expected == 0:
                self.assertIsNone(committed)
            else:
                scope = resolve_prefix(
                    committed, finish.call_args.args[0], service.config.limits
                )
                self.assertEqual(scope.requested.end_token, expected)
                self.assertEqual(scope.content.token_ids, tuple(range(expected)))
                self.assertLessEqual(expected, MAX_SCOPE_TOKENS)

    def test_hicache_runtime_profile_accepts_both_scheduler_modes(self):
        for disabled, tp_size in ((False, 1), (True, 1), (False, 2), (True, 4)):
            with self.subTest(disable_overlap_schedule=disabled, tp_size=tp_size):
                with get_context().override_server_args(
                    kv_control_config={"namespace": "ns"},
                    enable_hierarchical_cache=True,
                    disable_overlap_schedule=disabled,
                    tp_size=tp_size,
                ):
                    self.assertEqual(runtime_config().namespace, "ns")

    def test_runtime_profile_rejects_unsupported_config(self):
        for fields in [
            {"pp_size": 2},
            {"disable_radix_cache": True},
            {"enable_lmcache": True},
        ]:
            with self.subTest(fields=fields):
                with get_context().override_server_args(
                    kv_control_config={"namespace": "ns"},
                    enable_hierarchical_cache=True,
                    disable_overlap_schedule=False,
                    **fields,
                ):
                    with pytest.raises(ValueError):
                        runtime_config()


def namespace_request(salt=None, session=None):
    return Req(
        rid="r",
        origin_input_text=None,
        origin_input_ids=array("q", [1, 2, 3, 4]),
        sampling_params=SamplingParams(max_new_tokens=1),
        vocab_size=128,
        cache_salt=salt,
        session=session,
    )


def complete_request(cache, service, req):
    service.prepare_request(req)
    req.output_ids = array("q", [5])
    req.finished_reason = FINISH_LENGTH(1)
    cache.req_to_token_pool.alloc([req])
    slots = cache.token_to_kv_pool_allocator.alloc(5)
    cache.req_to_token_pool.write((req.kv.req_pool_idx, slice(0, 5)), slots)
    req.kv.kv_allocated_len = req.kv.kv_committed_len = 5
    release_kv_cache(req, cache, checkpoint=True)


def namespace_target(service, salt=None):
    return TokensTarget(
        namespace=service.config.namespace, token_ids=(1, 2, 3, 4, 5), cache_salt=salt
    )


class TestKVControlNamespaces(cache_fixtures.KVCacheTestCase):
    def test_ordinary_unhinted_request_and_independent_lookup_use_same_real_key(self):
        with service_owner(self, namespace_keys=True) as owner:
            salt = "tenant-a"
            from sglang.srt.managers.scheduler import Scheduler

            class TaggedReq(Req):
                pass

            class TaggedScheduler(Scheduler):
                def _create_generation_req(self, recv_req, *args, **kwargs):
                    return TaggedReq(*args, **kwargs)

            cache, service = owner
            cache_fixtures.insert(cache, [1, 2, 3, 4, 5], salt=salt)
            scheduler = Scheduler.__new__(TaggedScheduler)
            scheduler.kv_control_service = service
            req = scheduler._new_generation_req(
                SimpleNamespace(kv_hints=None, mm_inputs=None),
                rid="r",
                origin_input_text=None,
                origin_input_ids=array("q", [1, 2, 3, 4]),
                sampling_params=SamplingParams(max_new_tokens=1),
                vocab_size=128,
                cache_salt=salt,
            )
            assert isinstance(req, TaggedReq)
            complete_request(cache, service, req)
            assert service._request_sequence == 0
            assert not hasattr(req, "kv_control_context")
            assert req.cache_salt == encode_cache_salt("ns", salt)
            assert req.kv_control_logical_cache_salt == salt
            snapshot = service.control.executor.residency(
                resolve_prefix(
                    namespace_target(service, salt),
                    service.control.context("test"),
                    service.config.limits,
                )
            )
            assert (
                sum(
                    item.descriptor.end_token - item.descriptor.start_token
                    for item in snapshot.items
                )
                == 5
            )
            assert not snapshot.missing
            encoded_as_logical = service.control.executor.residency(
                resolve_prefix(
                    namespace_target(service, req.cache_salt),
                    service.control.context("test"),
                    service.config.limits,
                )
            )
            assert not encoded_as_logical.items and encoded_as_logical.missing
            cache.sanity_check()


class TestKVControlWeights(cache_fixtures.KVCacheTestCase):
    def test_mutations_rejected_before_touching_runner_or_cache(self):
        from sglang.srt.managers.scheduler import Scheduler

        with service_owner(self, namespace_keys=True) as (_, service):
            scheduler = Mock(kv_control_service=service)
            worker = Mock()
            flush = Mock()
            with get_parallel().override(tp_group=SimpleNamespace(cpu_group=None)):
                manager = SchedulerWeightUpdaterManager(
                    tp_worker=worker,
                    draft_worker=None,
                    memory_saver_adapter=None,
                    flush_cache=flush,
                    is_fully_idle=lambda: True,
                    scheduler=scheduler,
                )
            for method in (
                manager.begin_weight_update,
                manager.end_weight_update,
                manager.update_weights_from_disk,
                manager.update_weights_from_tensor,
                manager.update_weights_from_distributed,
                manager.update_weights_from_ipc,
                manager.release_memory_occupation,
                manager.resume_memory_occupation,
            ):
                with self.subTest(method=method.__name__):
                    self.assertFalse(method(SimpleNamespace()).success)
            self.assertFalse(
                manager.check_weights(
                    CheckWeightsReqInput(action="reset_tensors")
                ).success
            )
            self.assertFalse(
                Scheduler.handle_update_weight_version(
                    scheduler, UpdateWeightVersionReqInput(new_version="v2")
                ).success
            )
            self.assertEqual(worker.mock_calls, [])
            scheduler.record_weight_version_change.assert_not_called()
            flush.assert_not_called()
            self.assertIsNone(manager._session)
            self.assertEqual(service.capabilities()["weight_updates"], "unsupported")

    def test_disk_update_uses_normal_runner_when_control_disabled(self):
        worker = Mock()
        runner = Mock()
        runner.weight_updater.update_weights_from_disk.return_value = (True, "ok")
        worker.weight_update_runners.return_value = [("target", runner)]
        scheduler = Mock(kv_control_service=None)
        flush = Mock(return_value=True)
        with get_parallel().override(tp_group=SimpleNamespace(cpu_group=None)):
            manager = SchedulerWeightUpdaterManager(
                tp_worker=worker,
                draft_worker=None,
                memory_saver_adapter=None,
                flush_cache=flush,
                is_fully_idle=lambda: True,
                scheduler=scheduler,
            )
        result = manager.update_weights_from_disk(
            UpdateWeightFromDiskReqInput(model_path="local", weight_version="v2")
        )
        self.assertTrue(result.success)
        runner.weight_updater.update_weights_from_disk.assert_called_once_with(
            "local", None, recapture_cuda_graph=False
        )
        scheduler.record_weight_version_change.assert_called_once_with(new_version="v2")
        flush.assert_called_once()


if __name__ == "__main__":
    unittest.main()

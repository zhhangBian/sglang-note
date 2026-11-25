# SPDX-License-Identifier: Apache-2.0
"""CPU HiCache pools and controllable L2 transfer fences for KV control tests."""

import atexit
import json
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from unittest.mock import patch

import test_kv_control_cache as cache_fixtures
import torch

from sglang.srt.managers.kv_control.service import KVControlService, ServiceConfig
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.l2_transfer import TransferCompletion
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_context, get_parallel, get_server_args
from sglang.test.tree_slots import tree_slots_of


FULL = ComponentType.FULL


class HostRegistration:
    """CPU stand-in for the CUDA driver, with paired registration ownership."""

    def __init__(self):
        self.live = {}

    def cudaHostRegister(self, ptr, size, flags):
        assert ptr not in self.live, "Host range registered twice"
        self.live[ptr] = size
        return 0

    def cudaHostUnregister(self, ptr):
        assert ptr in self.live, "Unregistering an unowned host range"
        del self.live[ptr]
        return 0


class Fence:
    def __init__(self, ready=False):
        self.ready = ready

    def query(self):
        return self.ready

    def synchronize(self):
        assert self.ready, "Prematurely waited for an unfinished transfer"


class ControlledDMA:
    def __init__(self):
        self.pending = []
        self.submitted = []

    def submit(self, direction, transfers, on_layer_done=None):
        event = Fence()
        self.pending.append((direction, transfers, event, on_layer_done))
        self.submitted.append(
            (direction, sum((len(t.device_indices) for t in transfers)))
        )
        return TransferCompletion(Fence(True), event, False)

    def submit_device_to_host(self, transfers):
        return self.submit("save", transfers)

    def submit_host_to_device(
        self, transfers, *, transfer_layer_id_max, start_event=None, on_layer_done=None
    ):
        return self.submit("load", transfers, on_layer_done)

    def finish(self):
        direction, transfers, event, on_layer_done = self.pending.pop(0)
        for transfer in transfers:
            host, device = (transfer.host_pool, transfer.device_pool)
            for layer in range(device.layer_num):
                for h, d in (
                    (host.k_buffer[layer], device.get_key_buffer(layer)),
                    (host.v_buffer[layer], device.get_value_buffer(layer)),
                ):
                    if direction == "save":
                        h[transfer.host_indices] = d[transfer.device_indices]
                    else:
                        d[transfer.device_indices] = h[transfer.host_indices]
                if on_layer_done is not None:
                    on_layer_done(layer)
        event.ready = True


def seed(cache, tokens=(1, 2, 3, 4), value=7, *, salt=None):
    slots = cache.token_to_kv_pool_allocator.alloc(len(tokens))
    assert slots is not None
    pool = cache.token_to_kv_pool_allocator.get_kvcache()
    pool.k_buffer[0][slots] = value
    pool.v_buffer[0][slots] = value + 1
    executor = cache.kv_control_service.control.executor
    compatibility = executor._compatibility(
        cache_fixtures.target(tokens, cache_salt=salt)
    )
    for node, start, _, matched in executor._walk(tokens, compatibility):
        cd = node.component_data[ComponentType.FULL]
        old = cd.value if cd.value is not None else cd.host_value
        if old is not None:
            source = (
                pool
                if cd.value is not None
                else cache.host_pool_group.anchor_entry.host_pool
            )
            length = matched - start
            pool.k_buffer[0][slots[start:matched]] = source.k_buffer[0][old[:length]]
            pool.v_buffer[0][slots[start:matched]] = source.v_buffer[0][old[:length]]
    result = cache.insert(
        cache_fixtures.InsertParams(
            key=cache_fixtures.RadixKey(
                cache_fixtures.array("q", tokens), cache_salt=compatibility.cache_salt
            ),
            value=slots.to(torch.int64),
        )
    )
    return cache.tree_core.node_by_id(result.last_device_node)


@contextmanager
def hicache(
    case, config=None, *, storage=False, namespace_keys=False, write_policy="write_back"
):
    config = config or ServiceConfig(namespace="ns", engine_id="engine")
    storage = storage or namespace_keys
    storage_dir = TemporaryDirectory(prefix="sglang-kv-test-")
    case.addCleanup(storage_dir.cleanup)
    storage_args = (
        {
            "hicache_storage_backend": "file",
            "hicache_storage_backend_extra_config": json.dumps(
                {"file_storage_path": storage_dir.name}
            ),
        }
        if storage
        else {}
    )
    scope = get_context().override_server_args(
        device="cpu",
        enable_hierarchical_cache=True,
        hicache_ratio=2,
        hicache_write_policy=write_policy,
        hicache_mem_layout="layer_first",
        disable_overlap_schedule=True,
        **storage_args,
    )
    scope.install()
    parallel = get_parallel().override(
        tp_rank=0, tp_size=1, pp_rank=0, pp_size=1, attn_dcp_rank=0
    )
    parallel.__enter__()
    registrations = HostRegistration()
    driver = patch.object(torch.cuda, "cudart", return_value=registrations)
    driver.start()
    try:
        cache, _ = case.make(
            session=False, config=config.cache_config(), limits=config.limits
        )
        cache.reset()
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=cache.req_to_token_pool,
            token_to_kv_pool_allocator=cache.token_to_kv_pool_allocator,
            page_size=1,
            tree_slots=tree_slots_of((ComponentType.FULL,)),
            kv_control_config=config.cache_config(),
        )
        cache.init_hicache(get_server_args(), params)
        host_pool = cache.host_pool_group.anchor_entry.host_pool
        registered = host_pool.pin_memory
        if storage:
            # File I/O uses CPU staging tensors; only the backing pool is
            # registered with the simulated driver. Restore its ownership flag
            # before destroy so real cleanup releases those registrations.
            host_pool.pin_memory = False
        atexit.unregister(cache.shutdown)
        dma = ControlledDMA()
        cache.cache_controller.l2_transfer_engine = dma
        service = KVControlService(
            cache,
            config,
            clock=case.clock,
            namespace_keys=namespace_keys,
        )
        try:
            yield (case, cache, service, dma)
            assert not service.has_pending, "Test must retire every operation"
            assert not cache.ongoing_write_through and (not cache.ongoing_load_back)
        finally:
            for pending in tuple(service.control._pending.values()):
                service.control.cancel(pending.ref)
            service.progress()
            while dma.pending:
                dma.finish()
            service.progress()
            if storage:
                cache._storage_attachment.shutdown()
            host_pool.pin_memory = registered
            # The registered driver above is simulated even with all CUDA
            # devices hidden; exercise its matching unregister path too.
            with patch("sglang.srt.mem_cache.pool_host.base._is_cuda", True):
                cache.release_host_resources()
            assert not registrations.live, "Host registrations leaked during teardown"
    finally:
        driver.stop()
        parallel.__exit__(None, None, None)
        scope.restore()


def start_inference_load(cache, node):
    with patch.object(cache, "load_back_threshold", 1):
        assert cache.load_back(node.id)
    cache.ready_to_load_host_cache()


def start_inference_backup(cache, node):
    from sglang.srt.mem_cache.unified_cache.cache_action import BackupKV

    chain = []
    while node.parent is not None:
        chain.append(node.id)
        node = node.parent
    cache._execute_and_commit_kv_backup(BackupKV(node_ids=chain[::-1]))
    cache.flush_pending_backups()


def finish_inference_backup(cache, dma, *, persist=False):
    while dma.pending:
        dma.finish()
    if persist:
        cache.writing_check()
    else:
        # Keep the backup in host memory for tests that do not need storage.
        with patch.object(cache, "write_backup_storage"):
            cache.writing_check()


def backup_prefix(cache, dma, node, *, evict=False, persist=False):
    start_inference_backup(cache, node)
    finish_inference_backup(cache, dma, persist=persist)
    assert node.component_data[FULL].host_value is not None
    if evict:
        tracker = {FULL: 0}
        while node.parent is not None and cache.tree_core._is_device_leaf(node):
            assert cache._evict_device_leaf(node.id, tracker) is None
            node = node.parent

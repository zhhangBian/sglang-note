# SPDX-License-Identifier: Apache-2.0
"""Generation and configuration for the cache's single control owner."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sglang.srt.managers.kv_control.contracts import ControlError, Identifier, validated

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeNode


@dataclass(frozen=True)
class CacheControlConfig:
    namespace: str
    max_query_nodes: int = 4096
    max_result_targets: int = 4096
    max_prepared_prefixes: int = 128
    max_concurrent_storage_reads: int = 4

    def __post_init__(self):
        validated(self.namespace, Identifier)
        for name in (
            "max_prepared_prefixes",
            "max_query_nodes",
            "max_result_targets",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 1 << 24:
                raise ValueError(f"{name} must be a positive bounded integer")
        if (
            type(self.max_concurrent_storage_reads) is not int
            or not 1 <= self.max_concurrent_storage_reads <= 64
        ):
            raise ValueError(
                "max_concurrent_storage_reads must be an integer in [1, 64]"
            )


@dataclass(frozen=True)
class CopySelection:
    node: UnifiedTreeNode
    component: ComponentType
    start: int
    end: int


class CacheControlState:
    def __init__(self, config: CacheControlConfig):
        self.config = config
        self.clock = time.monotonic
        self.generation = 0
        self.host_priorities: dict[int, int] = {}
        self._binding: tuple[str, str] | None = None
        self._binding_generation = -1

    def bind(self, engine_id: str, engine_epoch: str):
        binding = validated(engine_id, Identifier), validated(engine_epoch, Identifier)
        if self._binding is not None:
            same_generation = self._binding_generation == self.generation
            if same_generation and binding != self._binding:
                raise ControlError(
                    "busy", "Reset the cache before changing its control owner or epoch"
                )
            if not same_generation and engine_epoch == self._binding[1]:
                raise ControlError(
                    "stale_epoch", "Cache reset requires a fresh control epoch"
                )
        self._binding, self._binding_generation = binding, self.generation

    def reset(self):
        self.generation += 1
        self.host_priorities.clear()


def control_owner(params):
    from threading import get_ident

    from sglang.srt.mem_cache.allocator.page_interleave import (
        page_interleave_shard_size,
    )

    reason = ""
    owner = get_ident() if params.kv_control_config is not None else None
    if params.kv_control_config is not None:
        if params.is_eagle or params.pp_size != 1 or params.attn_cp_size != 1:
            reason = "Speculative, pipeline and context parallel caches are unsupported"
        elif (
            params.mtp_draft_device_pools
            or params.mtp_draft_mamba_pools
            or params.enable_mamba_extra_buffer
            or params.enable_mamba_extra_buffer_lazy
            or params.decoder_swa_bounded_replay
            or not params.swa_gates_prefix_match
            or page_interleave_shard_size(params.token_to_kv_pool_allocator) > 1
        ):
            reason = "Auxiliary/draft/sharded cache layouts are unsupported"
    return owner, reason

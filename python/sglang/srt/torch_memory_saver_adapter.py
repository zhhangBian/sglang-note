import logging
import threading
import time
from abc import ABC
from contextlib import contextmanager, nullcontext

try:
    import torch_memory_saver

    _memory_saver = torch_memory_saver.torch_memory_saver
    import_error = None
except ImportError as e:
    import_error = e
    pass

logger = logging.getLogger(__name__)


# 适配器类，主要用于内存管理，尤其是在大模型KV缓存（Key-Value Cache）等高内存占用场景下
# 核心用法是通过with memory_saver_adapter.region上下文管理器，包裹住大块张量的分配操作，从而实现特定区域的内存分配策略
# 这里定义了相应的接口，真实的实现在_TorchMemorySaverAdapterReal中
class TorchMemorySaverAdapter(ABC):
    @staticmethod
    def create(enable: bool):
        if enable and import_error is not None:
            logger.warning(
                "enable_memory_saver is enabled, but "
                "torch-memory-saver is not installed. Please install it "
                "via `pip3 install torch-memory-saver`. "
            )
            raise import_error
        return (
            _TorchMemorySaverAdapterReal() if enable else _TorchMemorySaverAdapterNoop()
        )

    def check_validity(self, caller_name):
        if not self.enabled:
            logger.warning(
                f"`{caller_name}` will not save memory because torch_memory_saver is not enabled. "
                f"Potential causes: `enable_memory_saver` is false, or torch_memory_saver has installation issues."
            )

    def configure_subprocess(self):
        raise NotImplementedError

    # 将相应的张量都打上tag标记，后续可以对这个区域进行批量操作
    def region(self, tag: str):
        raise NotImplementedError

    # 暂停指定tag区域的内存分配（即后续在该tag下的分配会被忽略或延迟）
    def pause(self, tag: str):
        raise NotImplementedError

    # 恢复指定tag区域的内存分配
    def resume(self, tag: str):
        raise NotImplementedError

    @property
    def enabled(self):
        raise NotImplementedError


# 具有真实的内存管理功能的适配器
class _TorchMemorySaverAdapterReal(TorchMemorySaverAdapter):
    """Adapter for TorchMemorySaver with tag-based control"""

    def configure_subprocess(self):
        # torch_memory_saver的作用是给不同的张量分配相应的标签，进行显存的分区管理
        # configure_subprocess用于多进程/子进程场景下的内存管理配置，确保子进程继承或正确初始化内存管理策略
        return torch_memory_saver.configure_subprocess()

    def region(self, tag: str):
        return _memory_saver.region(tag=tag)

    def pause(self, tag: str):
        return _memory_saver.pause(tag=tag)

    def resume(self, tag: str):
        return _memory_saver.resume(tag=tag)

    @property
    def enabled(self):
        return _memory_saver is not None and _memory_saver.enabled


# 没有任何效果的适配器
class _TorchMemorySaverAdapterNoop(TorchMemorySaverAdapter):
    @contextmanager
    def configure_subprocess(self):
        yield

    @contextmanager
    def region(self, tag: str):
        yield

    def pause(self, tag: str):
        pass

    def resume(self, tag: str):
        pass

    @property
    def enabled(self):
        return False

"""Launch the inference server."""

import os
import sys

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    # 启动引擎，开始接收参数
    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)

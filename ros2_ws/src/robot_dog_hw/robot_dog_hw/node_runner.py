"""Common main() for all nodes: init, spin, and a cleanup that signals cannot interrupt.

SIGTERM raises KeyboardInterrupt like SIGINT. Once cleanup starts, SIGINT and SIGTERM are
ignored.
"""

from __future__ import annotations

import signal
import sys
from typing import Callable, Optional, Tuple, Type

import rclpy


def _sigterm_to_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def run_node(make_node: Callable, name: str,
             on_exit: Optional[Callable] = None,
             fatal: Tuple[Type[BaseException], ...] = (ValueError,),
             args=None) -> int:
    """Run ``make_node()`` until shutdown and return the process exit code.

    An exception in ``fatal`` (invalid configuration) is printed and gives exit code 2.
    ``on_exit(node)`` runs with signals ignored, before the node is destroyed.
    """
    signal.signal(signal.SIGTERM, _sigterm_to_keyboard_interrupt)
    rclpy.init(args=args)
    node = None
    code = 0
    try:
        node = make_node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except fatal as exc:
        print(f'[{name}] FATAL: refusing to start: {exc}', file=sys.stderr, flush=True)
        code = 2
    except Exception as exc:  # rclpy ExternalShutdownException on context shutdown
        if type(exc).__name__ != 'ExternalShutdownException':
            raise
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if node is not None:
            if on_exit is not None:
                try:
                    msg = on_exit(node)
                    if msg:
                        print(f'[{name}] {msg}', file=sys.stderr, flush=True)
                except Exception as exc:  # cleanup failure
                    print(f'[{name}] cleanup failed: {exc}', file=sys.stderr, flush=True)
            try:
                node.destroy_node()
            except Exception:
                pass
        rclpy.try_shutdown()
    return code

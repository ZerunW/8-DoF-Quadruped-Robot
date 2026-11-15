"""Compatibility shim: expose `smbus2` under the legacy module name `smbus`.

The ros-humble-bno055 driver imports ``SMBus`` from here.
"""

from smbus2 import *  # noqa: F401,F403
from smbus2 import SMBus  # noqa: F401

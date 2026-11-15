"""Minimal PCA9685 driver on smbus2, used only by ``servo_driver_node``.

Pulses come from ``servo_map.ServoLimiter``; ``write_us`` checks them again against the
configured window and refuses the whole frame if any is outside. Register addresses follow
the NXP PCA9685 datasheet.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Mapping, Optional

from robot_dog_hw.servo_map import PCA9685_COUNTS, PCA9685_FRAME_US, us_to_counts

MODE1 = 0x00
MODE2 = 0x01
LED0_ON_L = 0x06
ALL_LED_OFF_H = 0xFD
PRESCALE = 0xFE

MODE1_RESTART = 0x80
MODE1_AI = 0x20        # register auto-increment
MODE1_SLEEP = 0x10     # oscillator off
MODE1_ALLCALL = 0x01
MODE2_OUTDRV = 0x04    # totem-pole outputs
FULL_OFF = 0x10        # bit 4 of LEDn_OFF_H

NUM_CHANNELS = 16


def prescale_for(oscillator_hz: float, pwm_hz: float) -> int:
    """PRE_SCALE = round(osc / (4096 * f)) - 1."""
    value = int(math.floor(oscillator_hz / (PCA9685_COUNTS * pwm_hz) + 0.5)) - 1
    if not 3 <= value <= 255:
        raise ValueError(f'prescale {value} out of range 3..255 (osc={oscillator_hz}, f={pwm_hz})')
    return value


def build_block(pulses_us: Mapping[int, float], frame_us: int = PCA9685_FRAME_US) -> List[int]:
    """64 bytes for LED0_ON_L..LED15_OFF_H: ON=0, OFF=counts; unused channels full-off."""
    block = [0, 0, 0, FULL_OFF] * NUM_CHANNELS
    for ch, us in pulses_us.items():
        counts = us_to_counts(us, frame_us)
        if not 0 < counts < PCA9685_COUNTS:
            raise ValueError(f'channel {ch}: {us} us -> {counts} counts out of range')
        block[4 * ch + 0] = 0
        block[4 * ch + 1] = 0
        block[4 * ch + 2] = counts & 0xFF
        block[4 * ch + 3] = (counts >> 8) & 0x0F
    return block


class PCA9685:
    """PCA9685 on Linux i2c-dev via smbus2; the 65-byte block write uses ``i2c_rdwr``."""

    def __init__(self, bus: int = 1, address: int = 0x40, oscillator_hz: float = 27_000_000,
                 pwm_hz: float = 50.0, min_us: float = 500.0, max_us: float = 2500.0,
                 retries: int = 3, smbus_factory: Optional[Callable] = None):
        self.bus_id = bus
        self.address = address
        self.oscillator_hz = oscillator_hz
        self.pwm_hz = pwm_hz
        self.min_us = min_us
        self.max_us = max_us
        self.retries = retries
        self.prescale = prescale_for(oscillator_hz, pwm_hz)
        # Nominal frame period (us) for the pulse math.
        self.frame_us = int(round(1e6 / pwm_hz))
        self._smbus_factory = smbus_factory
        self._bus = None

    # -- low level -------------------------------------------------------------
    def _open(self):
        if self._bus is None:
            if self._smbus_factory is not None:
                self._bus = self._smbus_factory(self.bus_id)
            else:
                from smbus2 import SMBus
                self._bus = SMBus(self.bus_id)
        return self._bus

    def _retry(self, fn, *args):
        last = None
        for attempt in range(1 + self.retries):
            try:
                return fn(*args)
            except OSError as exc:  # NACK / arbitration lost / bus timeout
                last = exc
                time.sleep(0.001 * (attempt + 1))
        raise last

    def _write_reg(self, reg: int, value: int) -> None:
        self._retry(self._open().write_byte_data, self.address, reg, value & 0xFF)

    def _read_reg(self, reg: int) -> int:
        return self._retry(self._open().read_byte_data, self.address, reg)

    def _write_block(self, reg: int, data: List[int]) -> None:
        from smbus2 import i2c_msg
        msg = i2c_msg.write(self.address, [reg] + list(data))
        self._retry(self._open().i2c_rdwr, msg)

    # -- API ---------------------------------------------------------------------
    def begin(self) -> None:
        """Sleep, set PRE_SCALE for pwm_hz, wake, restart with auto-increment."""
        self._write_reg(MODE1, MODE1_SLEEP | MODE1_ALLCALL)      # oscillator off
        self._write_reg(PRESCALE, self.prescale)
        self._write_reg(MODE2, MODE2_OUTDRV)
        self._write_reg(MODE1, MODE1_AI | MODE1_ALLCALL)         # wake
        time.sleep(0.005)                                         # oscillator start
        self._write_reg(MODE1, MODE1_RESTART | MODE1_AI | MODE1_ALLCALL)   # 0xA1

    def write_us(self, pulses_us: Mapping[int, float]) -> None:
        """Write all channels in one 64-byte block; unused channels full-off."""
        for ch, us in pulses_us.items():
            if not isinstance(ch, int) or not 0 <= ch < NUM_CHANNELS:
                raise ValueError(f'invalid PCA9685 channel {ch!r}')
            if not isinstance(us, (int, float)) or not math.isfinite(us) \
                    or not self.min_us <= us <= self.max_us:
                raise ValueError(f'channel {ch}: pulse {us!r} us outside '
                                 f'[{self.min_us}, {self.max_us}] - refusing whole frame')
        self._write_block(LED0_ON_L, build_block(pulses_us, self.frame_us))

    def all_off(self) -> None:
        """Full-off on all 16 outputs: the servos go limp. Used only by an explicit relax."""
        self._write_reg(ALL_LED_OFF_H, FULL_OFF)

    def check_alive(self) -> bool:
        """True if the chip is awake with our prescale; False after a brown-out reset."""
        mode1 = self._read_reg(MODE1)
        prescale = self._read_reg(PRESCALE)
        return not (mode1 & MODE1_SLEEP) and prescale == self.prescale

    def close(self) -> None:
        if self._bus is not None:
            try:
                self._bus.close()
            finally:
                self._bus = None


class FakePCA9685(PCA9685):
    """Dry-run PCA9685: same checks, records every frame instead of touching I2C."""

    def __init__(self, *args, **kwargs):
        kwargs.pop('smbus_factory', None)
        super().__init__(*args, **kwargs)
        self.registers = bytearray(256)
        self.registers[MODE1] = MODE1_SLEEP | MODE1_ALLCALL   # power-on state
        self.frames: List[Dict[int, float]] = []
        self.blocks: List[List[int]] = []
        self.all_off_count = 0
        self.max_frames = 1000   # frames and blocks kept

    def _write_reg(self, reg: int, value: int) -> None:
        if reg == MODE1:
            value &= ~MODE1_RESTART  # RESTART self-clears
        self.registers[reg] = value & 0xFF

    def _read_reg(self, reg: int) -> int:
        return self.registers[reg]

    def _write_block(self, reg: int, data: List[int]) -> None:
        if not self.registers[MODE1] & MODE1_AI:
            raise OSError('auto-increment not enabled (begin() not called)')
        self.registers[reg:reg + len(data)] = bytes(data)
        self.blocks.append(list(data))
        if len(self.blocks) > self.max_frames:
            del self.blocks[0]

    def write_us(self, pulses_us: Mapping[int, float]) -> None:
        super().write_us(pulses_us)
        self.frames.append(dict(pulses_us))
        if len(self.frames) > self.max_frames:
            del self.frames[0]

    def all_off(self) -> None:
        super().all_off()
        self.all_off_count += 1

    def close(self) -> None:
        pass

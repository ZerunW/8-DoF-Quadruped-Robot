# Hardware

## I2C devices

Both devices sit on I2C bus 1 of the Raspberry Pi 5 at 100 kHz.

| Address | Device | Driver |
|---|---|---|
| 0x40 | PCA9685 PWM driver, 8 servos at 50 Hz | `robot_dog_hw/pca9685.py` (smbus2) |
| 0x28 | BNO055 IMU, IMUPLUS mode, axis placement `P7` | `ros-humble-bno055` |

The BNO055 driver imports `smbus`, which conda-forge does not package. `src/ext/smbus_compat`
installs a top-level `smbus` module that re-exports `smbus2`.

The stock BNO055 driver does not recover from I2C errors, so the bringup launch respawns it
and `imu_supervisor_node` kills it when `/bno055/imu` goes silent.

## Servos

The FT6335 turns a full 360°, and 500–2500 µs maps to 0–360°, so one servo unit is one
degree. Thigh ("top") and calf ("bottom") motors map to the kinematic motor angles `m1` and
`m2`.

| Index | PCA9685 channel | Label | Leg / motor |
|---|---|---|---|
| 0 | 3 | left_front_bottom | lf calf (m2) |
| 1 | 2 | left_front_top | lf thigh (m1) |
| 2 | 12 | left_back_bottom | lb calf |
| 3 | 13 | left_back_top | lb thigh |
| 4 | 4 | right_front_bottom | rf calf |
| 5 | 5 | right_front_top | rf thigh |
| 6 | 11 | right_back_bottom | rb calf |
| 7 | 10 | right_back_top | rb thigh |

Motor angle to servo angle: `s = m + 90` on the left legs and `s = 360 - (m + 90)` on the
mirrored right legs. The driver then clamps to `servo_min`/`servo_max`, adds `servo_offset`
and converts to a pulse. Channels, offsets and limits live in the `servo_driver` section of
`src/robot_dog_bringup/config/robot_dog.yaml`.

## IMU mount

The BNO055 board is mounted with its X axis pointing backward, Y left and Z down. Placement
`P7` remaps that to the REP-103 body frame. If the board is not flat in the chassis, record
its reading on a level body as `level_target_roll_deg` / `level_target_pitch_deg` (see
`calib_offsets --level-here` in [bringup.md](bringup.md)).

## Leg geometry

Each leg is a planar five-bar linkage with a bell crank. Link lengths come from the CAD:
thigh 110 mm, calf 110 mm with a 30 mm lever, fan arms 24 and 30 mm at 90°, crank 24 mm,
coupler 28.2 mm and connecting link 110 mm. The knees point backward. The hips sit at
x = ±91.5 mm and y = ±88.2 mm from the body center. The feet are TPU spheres with a 17.5 mm
radius. See [model.md](model.md).

## PS5 DualSense (optional)

The controller connects over Bluetooth. The kernel `hid-playstation` driver exposes it as a
joystick and `joy_node` publishes `/joy`. One-time setup on the Pi:

```bash
sudo apt-get install -y bluez
sudo sed -i '/^\[General\]/a ClassicBondedOnly=false' /etc/bluetooth/input.conf
sudo systemctl restart bluetooth
sudo modprobe uhid hid_playstation
printf 'uhid\nhid_playstation\n' | sudo tee /etc/modules-load.d/gamepad.conf
```

To pair, hold PS + Create until the light bar flashes quickly, then run
`robotdog> pixi run pair_gamepad`. The script scans, pairs, trusts and connects, then checks
for the kernel input device. It is safe to run again. After that, pressing PS reconnects.

Controls (hold L1 to move):

| Control | Action |
|---|---|
| L1 (hold) | deadman: the dog steps in place and the sticks drive it |
| Left / right stick Y | left / right side forward or backward (tank drive) |
| R1 / R2 (hold) | raise / lower the body, works without L1 |
| Cross | emergency stop |
| Options | stand, re-arms after an emergency stop |

Releasing L1, or losing `/joy` for 0.5 s, sends a single zero command and the dog stops. The
mapping is in `robot_dog_control/gamepad_mapping.py` and its parameters are in the
`gamepad_teleop` and `joy_node` sections of `robot_dog.yaml`.

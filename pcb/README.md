# PCB

Servo shield for the Raspberry Pi 5: a PCA9685 PWM driver with 16 servo outputs, battery
input, a 5 V buck-boost converter, an I2C level translator to the Pi and a USB to UART
bridge. The board is 120.7 x 118.3 mm with four copper layers (top, GND, PWR, bottom) and
parts on both sides.

```text
altium/servo_shield.schdoc   schematic
altium/servo_shield.pcbdoc   layout
bom.csv                      bill of materials, taken from the schematic
```

The design was drawn in EasyEDA Pro (JLCEDA) and exported in Altium ASCII format. The two
files open in Altium Designer. KiCad 7 cannot load the layout. To order boards or get a PDF
of the schematic, open the files in Altium Designer or import them into EasyEDA Pro and export
Gerber, drill, pick-and-place or PDF files from there.

## Power

| Net | Source | Feeds |
|---|---|---|
| `VCC` | XT60 input (U14) through the P-channel MOSFET Q1 | servo power pins, U10 |
| `+5V` | U10 (TPS63060, set to 5.0 V by R19/R20) through R29, or USB-C through fuse F1 | PCA9685, I2C connectors, CH340C |
| `VPI` | `+5V` through R30 (0 Ω) | Pi header pins 2 and 4 |
| `3V3` | Pi header pins 1 and 17 | Pi side of the level translator |

The servo rail is the battery voltage, not regulated. The PCA9685 and the I2C connectors
are powered from `+5V`.

## Raspberry Pi

U5 is the 2x20 header for the Pi. Only I2C bus 1 and power are used:

| Pi pin | Signal | Goes to |
|---|---|---|
| 3 (GPIO2, SDA) | `SDA2` | U8 (PCA9306), then `SDA` on the 5 V side |
| 5 (GPIO3, SCL) | `SCL2` | U8 (PCA9306), then `SCL` on the 5 V side |
| 1, 17 | `3V3` | U8 reference and pull-ups R2, R6 |
| 2, 4 | `VPI` | `+5V` through R30 |

The remaining GPIO pins are broken out to the 1x20 headers J1 and J2 and are not connected
to anything else on the board.

## Servos

The PCA9685 (U1) sits on the 5 V I2C bus. Its address is set by A0 to A5, each pulled low
by 10 kΩ. Leave SJ_A0 to SJ_A5 unfitted for address 0x40, the one the software uses. Each
output passes through a 220 Ω series resistor to a 4x3 pin header with the rows marked
PWM, PWR and GND:

| Header | Channels | Leg in the default configuration |
|---|---|---|
| U2 | PWM0 to PWM3 | left front (channels 2, 3) |
| U13 | PWM4 to PWM7 | right front (channels 4, 5) |
| U3 | PWM8 to PWM11 | right back (channels 10, 11) |
| U4 | PWM12 to PWM15 | left back (channels 12, 13) |

Channel numbers per joint are in [hardware.md](../ros2_ws/docs/hardware.md).

## IMU and other I2C devices

H6, H7, H8, H9 (1x4 pin headers) and U9 (4-pin PH 2.0) carry the 5 V I2C bus with the pin
order `+5V`, `GND`, `SCL`, `SDA`. The bus is pulled up to 5 V by R3 and R4. The BNO055
connects to one of them and shares the bus with the PCA9685.

## USB and serial

USB1 (Type-C) connects to a CH340C. `TXD`, `RXD` and `DTR` go to the 2x4 socket U12, and
`TXD`/`RXD` also reach D0/D1 of the Arduino UNO header position (H2 to H5) through R26 and
R27. The robot software does not use this part of the board.

## LEDs

| LED | Lit by |
|---|---|
| LED6 | `VCC`, servo rail |
| LED3 | `+5V` |
| LED2 | `VPI`, Pi 5 V |
| LED4, LED5 | `RXD`, `TXD` |

## Bill of materials

`bom.csv` lists all 94 schematic components with value, footprint, manufacturer part
number and LCSC number where the schematic has one. Generic 0603 passives carry a value
only. U6 is the Arduino UNO outline and is not a part. The footprints come from the
EasyEDA/LCSC library.

## License

MIT, see [LICENSE](../LICENSE).

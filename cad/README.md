# CAD

STEP files (AP242) of the printed parts, exported from Shapr3D. Units are millimetres. Every
file keeps its position in the robot assembly, so importing all of them into one document
reproduces the assembled robot without placing anything by hand.

```text
step/
  body.step          chassis and the eight servo holders (nine bodies)
  top_cover.step     three bodies: two side rails and one end piece
  leg_left/          the seven parts of one left leg
  leg_right/         the seven parts of one right leg
```

## Parts

| File | Per robot | Left / right | Material | Volume |
|---|---|---|---|---|
| `body.step` | 1 | | | 345.8 cm³ chassis, 6.6 to 7.0 cm³ per servo holder |
| `top_cover.step` | 1 | | | 19.2 cm³ |
| `upper_leg_sla.step` | 4 | 2 left, 2 right (mirrored) | SLA resin | 37.9 cm³ |
| `lower_leg.step` | 4 | 2 left, 2 right (mirrored) | | 13.0 cm³ |
| `rotating_link.step` | 4 | 2 left, 2 right (mirrored) | | 5.3 cm³ |
| `leg_connector.step` | 4 | same part on both sides | | 4.0 cm³ |
| `link.step` | 4 | same part on both sides | | 2.2 cm³ |
| `servo_connector_sla.step` | 4 | same part on both sides | SLA resin | 1.5 cm³ |
| `foot_flexible.step` | 4 | same part on both sides | TPU | 10.0 cm³ |

`leg_right/` is the mirror image of `leg_left/` about the plane x = -48.5 mm. The two legs
in this folder sit at one end of the body. The legs at the other end use the same parts,
shifted by -183 mm along Y (twice the 91.5 mm hip offset in
[hardware.md](../ros2_ws/docs/hardware.md)).

## Printing

- Files ending in `_sla` were designed for SLA resin printing.
- `foot_flexible.step` is printed in TPU. The foot is a sphere of 17.5 mm radius.
- The other parts have no material noted in the CAD.
- Slicers that read STEP (PrusaSlicer, OrcaSlicer, Bambu Studio) can open these files
  directly. Each part comes in at its assembly position and has to be laid flat on the bed
  in the slicer.

## Names used in the software

The URDF and the MuJoCo model use their own mesh names for the same parts
(`ros2_ws/src/robot_dog_description/meshes/`, in metres and in link frames):

| CAD file | Mesh | Role in the leg |
|---|---|---|
| `upper_leg_sla` | `thigh` | thigh, 110 mm |
| `lower_leg` | `calf` | calf, 110 mm with the 30 mm lever |
| `rotating_link` | `fan` | bell crank, 24 and 30 mm arms at 90° |
| `leg_connector` | `ab_link` | connecting link, 110 mm |
| `link` | `cd_link` | coupler, 28.2 mm |
| `servo_connector_sla` | `rod` | crank on the servo, 24 mm |
| `foot_flexible` | `foot` | foot |

## Not included

The servo models are left out because they are not part of this design. The robot uses
8 x Feetech FT6335, see [hardware.md](../ros2_ws/docs/hardware.md).

## License

MIT, see [LICENSE](../LICENSE).

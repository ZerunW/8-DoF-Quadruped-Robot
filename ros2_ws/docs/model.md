# Robot model and joint limits

## URDF

`robot_dog_description/urdf/robot_dog.urdf.xacro` reads the joint positions measured from the
CAD in `config/cad_frames.yaml` and the meshes in `meshes/`. Each closed five-bar leg is cut
into an open tree of six revolute joints, all about body +Y and all zero at the CAD pose
(m1 = m2 = 90°):

```
base_link -> thigh (O, motor m1) -> calf (K) -> foot (E, fixed)
base_link -> fan (O, passive)    -> ab_link (A)   loop closes at B on the calf
base_link -> rod (M, motor m2)   -> cd_link (C)   loop closes at D on the fan
```

`joint_state_from_servo` computes all six angles from `/servo_state` with the forward
kinematics and publishes `/joint_states` for the 3D view. After editing the URDF, regenerate
the MuJoCo model with `pixi run build_mjcf`.

## Joint region

The FT6335 turns a full 360°, so a leg is limited only by linkage closure and part contact.
The feasible (m1, m2) region was computed from the CAD meshes and is stored as
`gait_controller.ik_region_*` in `robot_dog.yaml`. IK never leaves it, and
`servo_driver.servo_min` / `servo_max` are its bounding box.

## Front/rear leg collision set

The front and rear legs of one side can touch when the front leg folds back over the rear
one. `robot_dog_description/doc/pair_sweep_<side>_final.json` lists every unsafe pair of
(rear pose, front pose) on a 2° grid of the joint region, computed from the CAD meshes with a
2 mm clearance. `robot_dog_control/unsafe_pairs.py` loads it for the startup envelope check
and the per-tick guard.

Both the region and the collision set belong to this exact mechanical design. If you change
the legs, recompute them for your CAD before running the robot.

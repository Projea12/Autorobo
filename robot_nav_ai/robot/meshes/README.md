# Mesh Files

Place robot mesh files here (.stl or .dae).

| File           | Link        | Notes                        |
|----------------|-------------|------------------------------|
| chassis.stl    | base_link   | Main chassis body            |
| wheel.stl      | left/right  | Shared wheel mesh            |
| lidar.stl      | lidar_link  | LiDAR sensor housing         |
| camera.stl     | camera_link | Camera module                |

To activate a mesh in robot.urdf, replace the primitive geometry with:
  <mesh filename="package://robot_nav_ai/robot/meshes/chassis.stl" scale="1 1 1"/>

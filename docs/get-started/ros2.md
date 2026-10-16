# ROS 2

The ROS 2 node runs the estimators on motion capture data and publishes the estimated states. The commands below run in the pixi `default` environment (`pixi shell`), which includes ROS 2 Kilted and [motion_capture_tracking](https://github.com/learnsyslab/motion_capture_tracking), see [Installation](installation.md).

## Motion capture

The node estimates the drones from their poses on `/tf`, where the `child_frame_id` is the drone name. Start the motion capture tracking with

```bash
ros2 launch motion_capture_tracking launch.py
```

The type and IP address of the motion capture system are set in the [configuration](https://github.com/learnsyslab/motion_capture_tracking/blob/ros2/motion_capture_tracking/config/cfg.yaml) of motion_capture_tracking.

## Estimator node

Start the estimator for a single drone with

```bash
python -m drone_estimators.ros_nodes.ros2_node --drone_name cf1
```

The node waits until it has received a first pose of every drone, initializes the estimators with it, and then runs at a fixed frequency. Add `--legacy` to use the legacy estimator instead of the UKF.

The settings of the estimators are read from a TOML file. By default, the node uses the [estimators.toml](https://github.com/learnsyslab/drone-estimators/blob/main/drone_estimators/ros_nodes/estimators.toml) that ships with the package:

```toml
--8<-- "drone_estimators/ros_nodes/estimators.toml"
```

To estimate several drones, add one `[estimatorN]` section per drone to your own copy of the file and pass it to the node. `--drone_name` and `--legacy` overwrite the settings of `estimator1`.

```bash
python -m drone_estimators.ros_nodes.ros2_node --settings path/to/estimators.toml
```

Estimators with identical settings (except for the drone name) run as one batched estimator.

!!! note "Localhost only"
    Since motion_capture_tracking may run on several computers in the same network, the node sets `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` and only communicates with nodes on the same machine.

### Topics

For each drone `<name>`, the node subscribes to

| Topic | Type | Description |
|---|---|---|
| `/tf` | `tf2_msgs/TFMessage` | Measured pose, the `child_frame_id` is the drone name |
| `/drones/<name>/command` | `std_msgs/Float64MultiArray` | Command sent to the drone: roll, pitch, yaw (deg) and thrust (PWM). Only used by the UKF |

and publishes

| Topic | Type | Description |
|---|---|---|
| `/drones/<name>/estimate/pose` | `geometry_msgs/PoseStamped` | Estimated position and orientation |
| `/drones/<name>/estimate/twist` | `geometry_msgs/TwistStamped` | Estimated linear and angular velocity |
| `/drones/<name>/estimate/forces` | `std_msgs/Float64MultiArray` | Estimated rotor velocities, if enabled |
| `/drones/<name>/estimate/wrench` | `geometry_msgs/WrenchStamped` | Estimated disturbance force and torque, if enabled |
| `/drones/<name>/estimate/marker_array` | `visualization_msgs/MarkerArray` | Markers to visualize the estimate in RViz |

### Calibration

If the markers are not mounted perfectly level, the measured orientation of the drone is tilted. With the drone standing on a level surface, call

```bash
ros2 service call /drones/cf1/calibration std_srvs/srv/Trigger
```

to store the current orientation as the level orientation. All following measurements are corrected by it. `/drones/cf1/remove_calibration` removes the calibration.

## Accessing the estimates

The `ROSConnector` gives non-blocking access to the latest estimates from your own Python code. It subscribes in separate processes and stores the data in shared memory, such that reading the data never waits for ROS.

```python
import rclpy

from drone_estimators.ros_nodes.ros2_connector import ROSConnector

if __name__ == "__main__":  # The connector spawns processes, so guard the entry point
    rclpy.init()
    connector = ROSConnector(estimator_names=["cf1"], tf_names=["cf1", "gate1"], timeout=5.0)
    pos, vel = connector.pos["cf1"], connector.vel["cf1"]  # Estimated state of cf1
    cf1_raw_pos = connector.pos_raw["cf1"]  # Raw pose of cf1 from /tf
    gate_pos = connector.pos_raw["gate1"]  # Raw pose of gate1 from /tf
    connector.close()
    rclpy.shutdown()
```

Objects in `estimator_names` provide the full estimated state (`pos`, `quat`, `vel`, `ang_vel`, `forces`, `disturbance`), objects in `tf_names` their raw pose from `/tf` (`pos_raw`, `quat_raw`). An object can be in both lists. Then, `is_outdated` is true if either its raw pose or its estimate is outdated. Since the estimator keeps publishing predictions when an object is occluded, add estimated objects to `tf_names` as well to detect lost tracking. With `timeout`, the connector waits until it has received data for all objects and raises a `TimeoutError` otherwise. The connector can also publish commands with `cmd_topic` and `publish_cmd`.

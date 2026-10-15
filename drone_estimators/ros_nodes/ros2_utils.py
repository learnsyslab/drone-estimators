"""This file contains some util functions for ROS 2 nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from geometry_msgs.msg import Point, PoseStamped, TransformStamped, TwistStamped, WrenchStamped
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray

if TYPE_CHECKING:
    from collections import defaultdict

    from std_msgs.msg import Header

    from drone_estimators._typing import Array  # To be changed to array_api_typing later
    from drone_estimators.structs.estimator_data import EstimatorData


def find_transform(
    transforms: list[TransformStamped], child_frame_id: str
) -> TransformStamped | None:
    """Finds the transform of a given child frame."""
    return next((tf for tf in transforms if tf.child_frame_id == child_frame_id), None)


def header2sec(header: Header) -> float:
    """Extracts the current time in seconds from a message header."""
    return float(header.stamp.sec + header.stamp.nanosec * 1e-9)


def tf2array(tf: TransformStamped) -> tuple[Header, Array, Array]:
    """Converts a Transform into numpy arrays."""
    pos = np.array(
        [tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z]
    )

    quat = np.array(
        [
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w,
        ]
    )

    return tf.header, pos, quat


def time2sec_nsec(t: float) -> tuple[int, int]:
    """Converts time as float into time as sec and nsec for ROS."""
    sec = int(t)
    nsec = int(1e9 * (t - sec))
    return sec, nsec


def create_pose(time_stamp: float, drone: str, pos: Array, quat: Array) -> PoseStamped:
    """Creates a PoseStamped from position and orientation."""
    pose = PoseStamped()
    sec, nsec = time2sec_nsec(time_stamp)
    pose.header.stamp.sec = sec
    pose.header.stamp.nanosec = nsec
    pose.header.frame_id = "world"
    pose.pose.position.x = pos[0]
    pose.pose.position.y = pos[1]
    pose.pose.position.z = pos[2]
    pose.pose.orientation.x = quat[0]
    pose.pose.orientation.y = quat[1]
    pose.pose.orientation.z = quat[2]
    pose.pose.orientation.w = quat[3]

    return pose


def create_twist(time_stamp: float, drone: str, vel: Array, ang_vel: Array) -> TwistStamped:
    """Creates a TwistStamped based on velocity and angular velocity."""
    twist = TwistStamped()
    sec, nsec = time2sec_nsec(time_stamp)
    twist.header.stamp.sec = sec
    twist.header.stamp.nanosec = nsec
    twist.header.frame_id = drone
    twist.twist.linear.x = vel[0]
    twist.twist.linear.y = vel[1]
    twist.twist.linear.z = vel[2]
    twist.twist.angular.x = ang_vel[0]
    twist.twist.angular.y = ang_vel[1]
    twist.twist.angular.z = ang_vel[2]

    return twist


def create_array(time_stamp: float, drone: str, data: Array | None) -> Float64MultiArray:
    """Creates a generic Float64Array with the length of data.

    If data is None, an array zeros of length 4 is created.
    """
    if data is not None:
        array = Float64MultiArray()
        array.data = list(data)
        return array
    else:
        array = Float64MultiArray()
        array.data = [0.0, 0.0, 0.0, 0.0]
        return array


def create_wrench(
    time_stamp: float, drone: str, force: Array | None, torque: Array | None
) -> WrenchStamped:
    """Creates a WrenchStamped based on force and torque.

    If force or torque is None, it gets set to zeros.
    """
    wrench = WrenchStamped()
    sec, nsec = time2sec_nsec(time_stamp)
    wrench.header.stamp.sec = sec
    wrench.header.stamp.nanosec = nsec
    wrench.header.frame_id = drone
    if force is not None:
        wrench.wrench.force.x = force[0]
        wrench.wrench.force.y = force[1]
        wrench.wrench.force.z = force[2]
    else:
        wrench.wrench.force.x = 0.0
        wrench.wrench.force.y = 0.0
        wrench.wrench.force.z = 0.0
    if torque is not None:
        wrench.wrench.torque.x = 0.0
        wrench.wrench.torque.y = 0.0
        wrench.wrench.torque.z = 0.0

    return wrench


def make_arrow_marker_from_vector(
    id: int,
    ns: str,
    start_pos: Array,
    vector: Array,
    color_rgba: Array = [1, 1, 1, 1],
    scale: float = 1.0,
) -> Marker:
    """TODO."""
    marker = Marker()
    marker.id = id
    marker.ns = ns
    marker.header.frame_id = "world"
    marker.type = Marker.ARROW
    marker.action = Marker.ADD

    # Arrow from start_pos to start_pos + vector * scale
    start = Point()
    start.x = start_pos[0]
    start.y = start_pos[1]
    start.z = start_pos[2]

    end = Point()
    end.x = start.x + vector[0] * scale
    end.y = start.y + vector[1] * scale
    end.z = start.z + vector[2] * scale

    marker.points = [start, end]

    marker.scale.x = 0.01  # shaft diameter
    marker.scale.y = 0.05  # head diameter
    marker.scale.z = 0.05  # head length

    marker.color.r = color_rgba[0]
    marker.color.g = color_rgba[1]
    marker.color.b = color_rgba[2]
    marker.color.a = color_rgba[3]
    if np.linalg.norm(vector) < 0.1:
        marker.color.a = 0

    return marker


def create_marker_array(
    time_stamp: float,
    drone: str,
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    force: Array | None,
    torque: Array | None,
) -> MarkerArray:
    """Creates a MarkerArray for visualization in RViz."""
    marker_array = MarkerArray()
    # print(f"{quat=}")

    ### Drone marker
    marker = Marker()
    sec, nsec = time2sec_nsec(time_stamp)
    marker.header.stamp.sec = sec
    marker.header.stamp.nanosec = nsec
    marker.header.frame_id = "world"
    marker.pose.position.x = pos[0]
    marker.pose.position.y = pos[1]
    marker.pose.position.z = pos[2]
    marker.pose.orientation.x = quat[0]
    marker.pose.orientation.y = quat[1]
    marker.pose.orientation.z = quat[2]
    marker.pose.orientation.w = quat[3]

    marker.ns = "drone"
    marker.id = 0
    marker.type = Marker.MESH_RESOURCE
    marker.action = Marker.ADD
    marker.scale.x = marker.scale.y = marker.scale.z = 1.0
    marker.color.r = marker.color.g = marker.color.b = 1.0
    marker.color.a = 1.0
    marker.mesh_resource = "package://motion_capture_tracking/meshes/crazyflie2.dae"
    marker.mesh_use_embedded_materials = True
    marker_array.markers.append(marker)

    ### Velocity Marker
    marker_array.markers.append(
        make_arrow_marker_from_vector(1, "twist", pos, vel, [1.0, 0.0, 0.0, 1.0])
    )

    ### Angular Velocity Marker
    rot = R.from_quat(quat)
    ang_vel = rot.apply(ang_vel)  # Rotating the vector same as the done to be displayed correctly
    marker_array.markers.append(
        make_arrow_marker_from_vector(2, "twist", pos, ang_vel, [1.0, 1.0, 0.0, 1.0])
    )

    ### Force Marker
    marker_array.markers.append(
        make_arrow_marker_from_vector(3, "wrench", pos, force * 5, [0.0, 0.0, 1.0, 1.0])
    )

    ### Torque Marker
    marker_array.markers.append(
        make_arrow_marker_from_vector(4, "wrench", pos, torque * 5, [1.0, 0.0, 1.0, 1.0])
    )

    return marker_array


def append_state(data: defaultdict[str, list], time: float, state: EstimatorData):
    """Appends each states data to the corresponding of list in the dictionary."""
    data["time"].append(time)
    data["pos"].append(state.pos)
    data["quat"].append(state.quat)
    data["vel"].append(state.vel)
    data["ang_vel"].append(state.ang_vel)
    data["covariance"].append(state.covariance)
    # The keys keep their old names, since plot.py and existing recordings use them
    if state.rotor_vel is not None:
        data["forces_motor"].append(state.rotor_vel)
    else:
        data["forces_motor"] = []
    if state.dist_f is not None:
        data["forces_dist"].append(state.dist_f)
    else:
        data["forces_dist"] = []
    if state.dist_t is not None:
        data["torques_dist"].append(state.dist_t)
    else:
        data["torques_dist"] = []


def append_measurement(
    data: defaultdict[str, list], time: float, pos: Array, quat: Array, command: Array | None = None
):
    """Appends each measurment data to the corresponding of list in the dictionary."""
    data["time"].append(time)
    data["pos"].append(pos)
    data["quat"].append(quat)
    if command is None:
        command = np.zeros_like(quat)
    data["command"].append(command)

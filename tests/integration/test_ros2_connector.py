"""Integration tests of the ROS connector."""

from __future__ import annotations

import os
from threading import Thread
from uuid import uuid4

import numpy as np
import pytest

# Keep the test traffic on this machine, see ros2_node.py
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, TwistStamped
from rclpy.executors import SingleThreadedExecutor
from tf2_msgs.msg import TFMessage

from drone_estimators.ros_nodes.ros2_connector import ROSConnector


def _transform(name: str, pos: list[float], quat: list[float]) -> TransformStamped:
    tf = TransformStamped()
    tf.child_frame_id = name
    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = pos
    r = tf.transform.rotation
    r.x, r.y, r.z, r.w = quat
    return tf


@pytest.mark.integration
def test_raw_and_estimated_poses():
    """Tests that estimated objects provide estimates and raw tf poses at the same time."""
    # Unique names, so that real tf data (e.g. from motion capture) cannot interfere
    tf_name, est_name = f"gate_{uuid4().hex[:8]}", f"cf_{uuid4().hex[:8]}"
    tf_pos, tf_quat = [1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]
    raw_pos, raw_quat = [4.0, 5.0, 6.0], [0.0, 0.0, 1.0, 0.0]
    est_pos, est_quat = [7.0, 8.0, 9.0], [1.0, 0.0, 0.0, 0.0]
    est_vel = [0.1, 0.2, 0.3]

    rclpy.init()
    node = rclpy.create_node("test_ros2_connector_" + uuid4().hex)
    tf_pub = node.create_publisher(TFMessage, "/tf", 10)
    pose_pub = node.create_publisher(PoseStamped, f"/drones/{est_name}/estimate/pose", 10)
    twist_pub = node.create_publisher(TwistStamped, f"/drones/{est_name}/estimate/twist", 10)

    def publish():
        tfs = [_transform(tf_name, tf_pos, tf_quat), _transform(est_name, raw_pos, raw_quat)]
        tf_pub.publish(TFMessage(transforms=tfs))
        pose = PoseStamped()
        p, q = pose.pose.position, pose.pose.orientation
        p.x, p.y, p.z = est_pos
        q.x, q.y, q.z, q.w = est_quat
        pose_pub.publish(pose)
        twist = TwistStamped()
        twist.twist.linear.x, twist.twist.linear.y, twist.twist.linear.z = est_vel
        twist_pub.publish(twist)

    node.create_timer(0.01, publish)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    connector = None
    try:
        connector = ROSConnector(tf_names=[tf_name], estimator_names=[est_name], timeout=10.0)
        # Objects tracked with tf only get the tf pose everywhere
        assert np.allclose(connector.pos[tf_name], tf_pos)
        assert np.allclose(connector.quat[tf_name], tf_quat)
        assert np.allclose(connector.pos_raw[tf_name], tf_pos)
        assert np.allclose(connector.quat_raw[tf_name], tf_quat)
        # Estimated objects get the estimate and additionally the raw tf pose
        assert np.allclose(connector.pos[est_name], est_pos)
        assert np.allclose(connector.quat[est_name], est_quat)
        assert np.allclose(connector.pos_raw[est_name], raw_pos)
        assert np.allclose(connector.quat_raw[est_name], raw_quat)
        assert np.allclose(connector.vel[est_name], est_vel)
        # Both kinds of objects have their own update times
        assert not np.isnan(connector.times[tf_name])
        assert not np.isnan(connector.times[est_name])
    finally:
        if connector is not None:
            connector.close()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

"""Integration tests of the ROS connector."""

from __future__ import annotations

import os
import time
from threading import Thread
from types import SimpleNamespace
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import pytest

# Keep the test traffic on this machine, see node.py
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"

pytest.importorskip("rclpy", reason="needs ROS 2")

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, TwistStamped
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage

from drone_estimators.ros import ROSConnector

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

GATE_POS, GATE_QUAT = [1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]
RAW_POS, RAW_QUAT = [4.0, 5.0, 6.0], [0.0, 0.0, 1.0, 0.0]
EST_POS, EST_QUAT = [7.0, 8.0, 9.0], [1.0, 0.0, 0.0, 0.0]
EST_VEL, EST_ROTOR_VEL = [0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7]


def _transform(name: str, pos: list[float], quat: list[float]) -> TransformStamped:
    tf = TransformStamped()
    tf.child_frame_id = name
    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = pos
    r = tf.transform.rotation
    r.x, r.y, r.z, r.w = quat
    return tf


@pytest.fixture
def fake() -> Iterator[SimpleNamespace]:
    """Publish the raw poses of a gate and a drone on /tf, and estimates of the drone."""
    # Unique names, so that real tf data (e.g. from motion capture) cannot interfere
    fake = SimpleNamespace(gate=f"gate_{uuid4().hex[:8]}", drone=f"cf_{uuid4().hex[:8]}")
    fake.drone_tracked = True
    rclpy.init()
    node = rclpy.create_node("test_connector_" + uuid4().hex)
    tf_pub = node.create_publisher(TFMessage, "/tf", 10)
    topic = f"/drones/{fake.drone}/estimate"
    pose_pub = node.create_publisher(PoseStamped, f"{topic}/pose", 10)
    twist_pub = node.create_publisher(TwistStamped, f"{topic}/twist", 10)
    rotor_vel_pub = node.create_publisher(Float64MultiArray, f"{topic}/rotor_vel", 10)

    def publish():
        tfs = [_transform(fake.gate, GATE_POS, GATE_QUAT)]
        if fake.drone_tracked:
            tfs.append(_transform(fake.drone, RAW_POS, RAW_QUAT))
        tf_pub.publish(TFMessage(transforms=tfs))
        pose = PoseStamped()
        p, q = pose.pose.position, pose.pose.orientation
        p.x, p.y, p.z = EST_POS
        q.x, q.y, q.z, q.w = EST_QUAT
        pose_pub.publish(pose)
        twist = TwistStamped()
        twist.twist.linear.x, twist.twist.linear.y, twist.twist.linear.z = EST_VEL
        twist_pub.publish(twist)
        rotor_vel_pub.publish(Float64MultiArray(data=EST_ROTOR_VEL))

    node.create_timer(0.01, publish)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    Thread(target=executor.spin, daemon=True).start()
    yield fake
    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()


@pytest.fixture
def connect(fake: SimpleNamespace) -> Iterator[Callable[..., ROSConnector]]:
    """Create connectors that are closed after the test."""
    connectors = []

    def connect(**kwargs: list[str]) -> ROSConnector:
        connectors.append(ROSConnector(timeout=10.0, **kwargs))
        return connectors[-1]

    yield connect
    for connector in connectors:
        connector.close()


@pytest.mark.integration
def test_tf_names(fake: SimpleNamespace, connect: Callable[..., ROSConnector]):
    """Tests that objects in tf_names provide their raw poses."""
    connector = connect(tf_names=[fake.gate])
    assert np.allclose(connector.tf_pos[fake.gate], GATE_POS)
    assert np.allclose(connector.tf_quat[fake.gate], GATE_QUAT)
    assert not np.isnan(connector.tf_time[fake.gate])
    assert fake.gate not in connector.est_pos


@pytest.mark.integration
def test_estimator_names(fake: SimpleNamespace, connect: Callable[..., ROSConnector]):
    """Tests that objects in estimator_names provide their estimates."""
    connector = connect(estimator_names=[fake.drone])
    time.sleep(0.2)  # The timeout does not wait for rotor velocities
    assert np.allclose(connector.est_pos[fake.drone], EST_POS)
    assert np.allclose(connector.est_quat[fake.drone], EST_QUAT)
    assert np.allclose(connector.est_vel[fake.drone], EST_VEL)
    assert np.allclose(connector.est_rotor_vel[fake.drone], EST_ROTOR_VEL)
    assert not np.isnan(connector.est_time[fake.drone])
    assert fake.drone not in connector.tf_pos


@pytest.mark.integration
def test_tf_and_estimator_names(fake: SimpleNamespace, connect: Callable[..., ROSConnector]):
    """Tests that objects can provide their raw poses and estimates at the same time."""
    connector = connect(tf_names=[fake.drone], estimator_names=[fake.drone])
    assert np.allclose(connector.tf_pos[fake.drone], RAW_POS)
    assert np.allclose(connector.tf_quat[fake.drone], RAW_QUAT)
    assert np.allclose(connector.est_pos[fake.drone], EST_POS)
    assert np.allclose(connector.est_quat[fake.drone], EST_QUAT)


@pytest.mark.integration
def test_is_outdated(fake: SimpleNamespace, connect: Callable[..., ROSConnector]):
    """Tests that objects are outdated if either their raw pose or their estimate is."""
    connector = connect(tf_names=[fake.gate, fake.drone], estimator_names=[fake.drone])
    assert not any(connector.is_outdated.values())
    fake.drone_tracked = False  # Like the estimator node, the fake keeps publishing estimates
    time.sleep(0.5)
    assert connector.is_outdated[fake.drone]
    assert not connector.is_outdated[fake.gate]

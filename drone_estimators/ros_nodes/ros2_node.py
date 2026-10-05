"""This file contains the estimator node for ROS 2.

Essentially this file provides a wrapper for the estimators to be used with ROS data. The estimates get published to ROS as well.

TODO Subscribed and published topics...
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import pickle
import signal
import time
from collections import defaultdict, deque
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

# Since motion_capture_tracking might be running on multiple PCs simultaneously,
# the same /tf topics get published multiple times to the network. This is
# unintended behavior. Setting it to LOCALHOST will block publishing to the
# network. We also set it here in the estimators s.t. they also don't publish
# to the network. Usually, the estimators get started after
# motion_capture_tracking, but in case the estimators get started first, we
# get consistent behavior.
# Note: All ros nodes started after this line will also only publish locally!
# Identical console command: ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST"
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"

import jax
import numpy as np
import rclpy
import toml

# Message types: https://docs.ros2.org/foxy/api/geometry_msgs/index-msg.html
from crazyflow.control import load_params as load_controller_params
from crazyflow.control.transform import pwm2force
from crazyflow.drones import Drone
from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped
from munch import Munch, munchify
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import MarkerArray

from drone_estimators.estimator_kalman import KalmanFilter
from drone_estimators.estimator_legacy import StateEstimator
from drone_estimators.ros_nodes.ros2_utils import (
    append_measurement,
    append_state,
    create_array,
    create_marker_array,
    create_pose,
    create_twist,
    create_wrench,
    find_transform,
    tf2array,
)

# The jitted estimators need 64 bit precision. The numpy estimators are not affected. Since this
# runs on import, it also applies to the spawned estimator processes.
jax.config.update("jax_enable_x64", True)

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import SynchronizedArray
    from multiprocessing.synchronize import Barrier, Event

    from numpy.typing import NDArray


def _view(buffer: SynchronizedArray, n_drones: int) -> NDArray:
    """View a shared buffer as numpy array with one row per drone (without copying or locking)."""
    return np.frombuffer(buffer.get_obj(), dtype=np.float64).reshape(n_drones, -1)


class MPEstimator:
    """This class starts a batched estimator for all drones in settings.drone_names and the necessary subscribers and publishers.

    Note: Since all results are published individually, one might want to
    synchronize on the receiving end. For that see
    https://docs.ros.org/en/rolling/p/flex_sync/
    https://github.com/ros2/message_filters

    In the future, a custom message type might be advantageous. However,
    to keep it simple, we use standard message types.
    """

    def __init__(self, settings: Munch):
        """TODO."""
        self.settings = settings
        self.drone_names = settings.drone_names
        n_drones = len(self.drone_names)

        ctx = mp.get_context("spawn")
        self._shutdown = ctx.Event()
        self._publish_update = ctx.Event()
        startup = ctx.Barrier(3)  # Main process, _subscriber_loop, _publisher_loop

        # Logger setup
        self.logger = logging.getLogger("ESTIMATOR" + "_" + "_".join(self.drone_names))
        self.logger.setLevel(logging.INFO)
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # Create formatter with [name][LEVEL] prefix
        formatter = logging.Formatter("[%(name)s][%(levelname)s] %(message)s")
        # Add formatter to handler
        console_handler.setFormatter(formatter)
        # Add handler to logger (only once)
        if not self.logger.hasHandlers():
            self.logger.addHandler(console_handler)
        # All buffers contain one row per drone, see _view.
        # We allocate a buffer that is used to store the most recent tf message from the tf
        # subscriber. The message contains four fields: The first element is the number of messages
        # since the array was last cleared. This allows us to throw warnings in case the estimator
        # loop cannot keep up. The second field is the timestamp in s, the third is the position
        # (3d) and the fourth is the quaternion (4d)
        self._tf_msg_buffer = ctx.Array("d", [0.0] * (1 + 1 + 3 + 4) * n_drones)
        # Allocate the command subscriber buffer
        cmd_dim = 4
        self._cmd_msg_buffer = ctx.Array("d", [0.0] * (1 + 1 + cmd_dim) * n_drones)
        # Estimated state publisher buffers
        self._pose_buffer = ctx.Array("d", [0.0] * 7 * n_drones)
        self._twist_buffer = ctx.Array("d", [0.0] * 6 * n_drones)
        self._forces_buffer = ctx.Array("d", [0.0] * 4 * n_drones)  # Motor forces, optional
        self._wrench_buffer = ctx.Array("d", [0.0] * 6 * n_drones)  # External wrench, optional

        args = (
            self.drone_names,
            self._tf_msg_buffer,
            self._cmd_msg_buffer,
            startup,
            self._shutdown,
        )
        self._sub_process = ctx.Process(target=self._subscriber_loop, args=args)

        args = (
            self.drone_names,
            self._pose_buffer,
            self._twist_buffer,
            self._forces_buffer,
            self._wrench_buffer,
            self._publish_update,
            startup,
            self._shutdown,
        )
        self._pub_process = ctx.Process(target=self._publisher_loop, args=args)

        self._sub_process.start()
        self._pub_process.start()
        startup.wait(10.0)
        self.logger.info("Subscriber and Publisher process started.")

    def _init_estimator(self):
        self.input_needed = False
        self.initial_observation = None

        self.time_stamp_last_prediction = np.zeros(len(self.drone_names))
        self.time_stamp_last_correction = np.zeros(len(self.drone_names))
        self.perf_timings = deque(maxlen=5000)

        self.frequency = self.settings.frequency  # Hz # TODO get from vicon frequency

        self.current_header = None
        self.current_state = None

        # This is for storing the data (DEBUG_SAVE_DATA), one dict per drone
        self.data_meas = [
            defaultdict(list) for _ in self.drone_names
        ]  # {"time": [], "pos": [], "quat": [], "command": []}
        self.data_est = [defaultdict(list) for _ in self.drone_names]

        batch_shape, jit = (len(self.drone_names),), self.settings.jit_compile
        match self.settings.estimator_type:
            case "legacy":
                self.estimator = StateEstimator(
                    (0.0001, 0.007, 0.09, 0.005, 0.07), batch_shape=batch_shape, jit=jit
                )
                if (
                    self.settings.estimate_rotor_vel
                    or self.settings.estimate_dist_f
                    or self.settings.estimate_dist_t
                ):
                    self.logger.warning(
                        "Legacy estimator does not support force or torque estimation!"
                    )
            case "ukf":
                self.input_needed = True
                # The commands are converted from PWM to N like the onboard controller does
                drone = Drone(self.settings.drone_config)
                self.params = load_controller_params("mellinger", drone)["core"]
                self.estimator = KalmanFilter(
                    dt=1 / self.frequency,
                    model=self.settings.dynamics_model,
                    config=self.settings.drone_config,
                    estimate_rotor_vel=self.settings.estimate_rotor_vel,
                    estimate_dist_f=self.settings.estimate_dist_f,
                    estimate_dist_t=self.settings.estimate_dist_t,
                    batch_shape=batch_shape,
                    jit=jit,
                )
            case _:
                raise NotImplementedError(
                    f"Estimator type {self.settings.estimator_type} not implemented."
                )

        # Initialization, waits until all drones have been measured
        self.logger.info("Waiting for initial measurement.")
        tf_msg_buffer = _view(self._tf_msg_buffer, len(self.drone_names))
        while not self._shutdown.is_set():
            with self._tf_msg_buffer.get_lock():
                data = tf_msg_buffer.copy()
                tf_msg_buffer[:, 0] = 0
            n_tf_msg, tf_timestamp, pos, quat = data[:, 0], data[:, 1], data[:, 2:5], data[:, 5:]

            if np.all(n_tf_msg >= 1):
                self.time_stamp_last_prediction = tf_timestamp
                self.time_stamp_last_correction = tf_timestamp
                self.estimator.set_state(pos, quat)
                self.logger.info("Initialized pos and quat.")
                break

            time.sleep(0.5)

        self.logger.info(f"Started estimator (process {os.getpid()})")

    def run(self):
        """Main estimator loop."""
        self._init_estimator()  # done here such that errors can be raised properly

        names = np.array(self.drone_names)
        n_drones = len(self.drone_names)
        tf_msg_buffer = _view(self._tf_msg_buffer, n_drones)
        cmd_msg_buffer = _view(self._cmd_msg_buffer, n_drones)
        pose_buffer = _view(self._pose_buffer, n_drones)
        twist_buffer = _view(self._twist_buffer, n_drones)
        forces_buffer = _view(self._forces_buffer, n_drones)
        wrench_buffer = _view(self._wrench_buffer, n_drones)

        k = 0
        global_time = time.perf_counter()

        try:
            # self.logger.info(f"{self.estimator.data.pos=}")
            # Estimation loop
            while not self._shutdown.is_set():
                loop_start_time = time.time()

                with self._tf_msg_buffer.get_lock():
                    data = tf_msg_buffer.copy()
                    tf_msg_buffer[:, 0] = 0
                n_tf_msg, tf_timestamp, pos, quat = (
                    data[:, 0],
                    data[:, 1],
                    data[:, 2:5],
                    data[:, 5:],
                )
                with self._cmd_msg_buffer.get_lock():
                    data = cmd_msg_buffer.copy()
                    cmd_msg_buffer[:, 0] = 0
                n_cmd_messages, cmd_timestep, cmd = data[:, 0], data[:, 1], data[:, 2:]
                has_tf, has_cmd = n_tf_msg >= 1, n_cmd_messages >= 1

                outdated = (cmd_timestep < tf_timestamp - 1) & (cmd_timestep > 0)
                if np.any(outdated) and self.input_needed:
                    self.logger.warning(
                        f"Last command of {names[outdated]} is older than 1s. Assuming zeros as input."
                    )
                    u = np.array(self.estimator.data.u)
                    u[outdated] = 0.0
                    self.estimator.set_input(u)

                if np.any(has_cmd) and self.input_needed:
                    # The command is as it is sent to the drone, meaning for attitude interface:
                    # roll (deg), pitch (deg), yaw (deg), thrust (PWM)
                    # All the models run with rad and N, so we need to convert the RPYT command
                    cmd[..., -1] = pwm2force(
                        cmd[..., -1], self.params["thrust_max"] * 4, self.params["pwm_max"]
                    )
                    cmd[..., :-1] = np.deg2rad(cmd[..., :-1])
                    u = np.array(self.estimator.data.u)
                    u[has_cmd] = cmd[has_cmd]
                    self.estimator.set_input(u)  # TODO # compare times?

                if np.any(n_tf_msg > 2):
                    self.logger.warning(
                        f"Dropping tf messages of {names[n_tf_msg > 2]} because estimator loop can't keep up"
                    )
                    # TODO check for frequencies. If Vicon is running at higher frequency, of course estimator cant keep up

                if np.any(has_tf):
                    # Drones without a measurement have dt = 0 and are neither predicted nor corrected
                    dt = np.where(has_tf, tf_timestamp - self.time_stamp_last_prediction, 0.0)
                    self.time_stamp_last_prediction = np.where(
                        has_tf, tf_timestamp, self.time_stamp_last_prediction
                    )

                    self.estimator.predict(dt)
                    self.estimator.correct(pos, quat, mask=has_tf)

                    # estimator_data = self.estimator.step(pos, quat, dt)

                time_stamp_now = time.time()
                dt = time_stamp_now - self.time_stamp_last_prediction
                self.time_stamp_last_prediction = np.full(n_drones, time_stamp_now)
                estimator_data = self.estimator.predict(dt)

                # Giving new estimate to publisher
                with self._pose_buffer.get_lock():
                    pose_buffer[:, :3] = estimator_data.pos
                    pose_buffer[:, 3:] = estimator_data.quat
                with self._twist_buffer.get_lock():
                    twist_buffer[:, :3] = estimator_data.vel
                    twist_buffer[:, 3:] = estimator_data.ang_vel
                if estimator_data.rotor_vel is not None:
                    with self._forces_buffer.get_lock():
                        forces_buffer[:] = estimator_data.rotor_vel
                if estimator_data.dist_f is not None:
                    with self._wrench_buffer.get_lock():
                        wrench_buffer[:, :3] = estimator_data.dist_f
                        if estimator_data.dist_t is not None:
                            wrench_buffer[:, 3:] = estimator_data.dist_t
                self._publish_update.set()

                if self.settings.save_data:
                    estimator_data = jax.tree.map(np.asarray, estimator_data)
                    for i in range(n_drones):
                        estimate_i = jax.tree.map(lambda x: x[i], estimator_data)
                        append_state(self.data_est[i], time_stamp_now, estimate_i)
                        if has_tf[i]:
                            if has_cmd[i] and self.input_needed:
                                append_measurement(
                                    self.data_meas[i], tf_timestamp[i], pos[i], quat[i], cmd[i]
                                )
                            if not self.input_needed:
                                append_measurement(
                                    self.data_meas[i], tf_timestamp[i], pos[i], quat[i], None
                                )

                # if k % 100 == 0:
                #     self.logger.info(f"{estimator_data.dist_f=}")

                remaining = (
                    (1 / self.frequency) - (time.time() - loop_start_time) - 1.25 * 1e-4
                )  # TODO remove magic number, replace with "controller"
                if k % 1000 == 999:
                    self.logger.info(f"Freq: {k / (time.perf_counter() - global_time)}")
                k += 1
                if remaining > 0:
                    time.sleep(remaining)
        except KeyboardInterrupt:
            self._shutdown.set()

    @staticmethod
    def _subscriber_loop(
        drone_names: list[str],
        _tf_msg_buffer: SynchronizedArray,
        _cmd_msg_buffer: SynchronizedArray,
        startup: Barrier,
        shutdown: Event,
    ):
        rclpy.init()
        node = rclpy.create_node("estimator_sub_" + "_".join(drone_names))
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        signal.signal(signal.SIGINT, lambda c, _: shutdown.set())  # Gracefully handle Ctrl-C

        tf_msg_buffer = _view(_tf_msg_buffer, len(drone_names))
        cmd_msg_buffer = _view(_cmd_msg_buffer, len(drone_names))
        # last_quat = np.array([0.0, 0.0, 0.0, 1.0])
        # calibration_quat = np.array([0.0, 0.0, 0.0, 1.0])
        last_rot = [R.from_quat(np.array([0.0, 0.0, 0.0, 1.0])) for _ in drone_names]
        calibration_rot = [R.from_quat(np.array([0.0, 0.0, 0.0, 1.0])) for _ in drone_names]

        def tf_callback(msg: TFMessage):
            missing_tfs = []
            for i, drone_name in enumerate(drone_names):
                tf = find_transform(msg.transforms, drone_name)
                if tf is None:
                    missing_tfs.append(drone_name)
                    continue
                _, pos, quat = tf2array(tf)
                # We do not use the ros header time here because we need to establish an ordering
                # between observations and the model loop. We are not certain that the ros time
                # stamp is comparable to the system time. Therefore, we create a new timestamp
                # using the os time.
                # TODO: Subtract a constant time to account for [Vicon -> ros2 pub -> ros2 sub] delay.
                time_stamp = time.time()
                last_rot[i] = R.from_quat(quat)
                with _tf_msg_buffer.get_lock():
                    tf_msg_buffer[i, 0] += 1
                    tf_msg_buffer[i, 1] = time_stamp
                    tf_msg_buffer[i, 2:5] = pos
                    tf_msg_buffer[i, 5:9] = (calibration_rot[i].inv() * last_rot[i]).as_quat()
            # Warn once for all drones, since the throttling is shared by all drones
            if missing_tfs:
                node.get_logger().warning(
                    f"Drones {missing_tfs} could not have been found. Occluded?",
                    throttle_duration_sec=0.5,
                )

        def cmd_callback(msg: Float64MultiArray, i: int):
            # The command is as it is sent to the drone, meaning for attitude interface:
            # roll (deg), pitch (deg), yaw (deg), thrust (PWM)
            with _cmd_msg_buffer.get_lock():
                cmd_msg_buffer[i, 0] += 1
                cmd_msg_buffer[i, 1] = time.time()
                cmd_msg_buffer[i, 2:] = msg.data[:4]  # slice as cmd_dim

        def calibration_callback(
            request: Trigger.Request, response: Trigger.Response, i: int
        ) -> Trigger.Response:
            rpy = last_rot[i].as_euler("xyz", degrees=True)
            max_angle = 20  # degrees
            if np.any(rpy > max_angle):
                node.get_logger().warning("Calibration failed.")
                response.success = False
                response.message = "Pose could not be calibrated, deck tilted too much."
                return response

            node.get_logger().info("Calibration successful.")
            calibration_rot[i] = last_rot[i]
            response.success = True
            response.message = "Pose calibrated successfully."
            return response

        def remove_calibration_callback(
            request: Trigger.Request, response: Trigger.Response, i: int
        ) -> Trigger.Response:
            node.get_logger().info("Calibration deleted successfully.")
            calibration_rot[i] = R.from_quat(np.array([0.0, 0.0, 0.0, 1.0]))
            response.success = True
            response.message = "Calibration deleted successfully."
            return response

        sub_tf = node.create_subscription(TFMessage, "/tf", tf_callback, qos_profile=qos_profile)
        subs = []
        for i, drone_name in enumerate(drone_names):
            sub_cmd = node.create_subscription(
                Float64MultiArray,
                f"/drones/{drone_name}/command",
                partial(cmd_callback, i=i),
                qos_profile=qos_profile,
            )
            sub_calib = node.create_service(
                Trigger, f"/drones/{drone_name}/calibration", partial(calibration_callback, i=i)
            )
            sub_remove_calib = node.create_service(
                Trigger,
                f"/drones/{drone_name}/remove_calibration",
                partial(remove_calibration_callback, i=i),
            )
            subs += [sub_cmd, sub_calib, sub_remove_calib]
        startup.wait(10.0)  # Register this process as ready for startup barrier

        while not shutdown.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
        sub_tf.destroy()
        for sub in subs:
            sub.destroy()
        node.destroy_node()

    @staticmethod
    def _publisher_loop(
        drone_names: list[str],
        pose_buffer: SynchronizedArray,
        twist_buffer: SynchronizedArray,
        forces_buffer: SynchronizedArray,
        wrench_buffer: SynchronizedArray,
        update: Event,
        startup: Barrier,
        shutdown: Event,
    ):
        rclpy.init()
        node = rclpy.create_node("estimator_sub_" + "_".join(drone_names))
        # TODO check if pubs are actually needed?
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        pubs_pose, pubs_twist, pubs_forces, pubs_wrench, pubs_markers = [], [], [], [], []
        for drone_name in drone_names:
            # pos, quat
            pub_pose = node.create_publisher(
                PoseStamped, f"/drones/{drone_name}/estimate/pose", qos_profile=qos_profile
            )
            # vel, ang_vel
            pub_twist = node.create_publisher(
                TwistStamped, f"/drones/{drone_name}/estimate/twist", qos_profile=qos_profile
            )
            # f_motors
            pub_forces = node.create_publisher(
                Float64MultiArray, f"/drones/{drone_name}/estimate/forces", qos_profile=qos_profile
            )
            # f_dis, t_dis
            pub_wrench = node.create_publisher(
                WrenchStamped, f"/drones/{drone_name}/estimate/wrench", qos_profile=qos_profile
            )
            # marker (only for rviz)
            pub_markers = node.create_publisher(
                MarkerArray, f"/drones/{drone_name}/estimate/marker_array", qos_profile=qos_profile
            )
            pubs_pose.append(pub_pose)
            pubs_twist.append(pub_twist)
            pubs_forces.append(pub_forces)
            pubs_wrench.append(pub_wrench)
            pubs_markers.append(pub_markers)
        signal.signal(signal.SIGINT, lambda c, _: shutdown.set())  # Gracefully handle Ctrl-C
        startup.wait(10.0)  # Register this process as ready for startup barrier

        # TODO clear arrays???
        while not shutdown.is_set():
            if not update.wait(0.5):
                continue
            time_stamp = time.time()
            update.clear()

            n_drones = len(drone_names)
            poses = np.asarray(pose_buffer, dtype=np.float64, copy=True).reshape(n_drones, -1)
            twists = np.asarray(twist_buffer, dtype=np.float64, copy=True).reshape(n_drones, -1)
            forces_all = np.asarray(forces_buffer, dtype=np.float64, copy=True).reshape(
                n_drones, -1
            )
            wrenches = np.asarray(wrench_buffer, dtype=np.float64, copy=True).reshape(n_drones, -1)

            for i, drone_name in enumerate(drone_names):
                pose = poses[i]
                pose_stamped = create_pose(time_stamp, drone_name, pose[:3], pose[3:])
                pubs_pose[i].publish(pose_stamped)

                twist = twists[i]
                twist_stamped = create_twist(time_stamp, drone_name, twist[:3], twist[3:])
                pubs_twist[i].publish(twist_stamped)

                forces = forces_all[i]
                # This type doesn't have a stamp!
                forces_array = create_array(time_stamp, drone_name, forces)
                pubs_forces[i].publish(forces_array)

                wrench = wrenches[i]
                wrench_stamped = create_wrench(time_stamp, drone_name, wrench[:3], wrench[3:])
                pubs_wrench[i].publish(wrench_stamped)

                markers = create_marker_array(
                    time_stamp,
                    drone_name,
                    pose[:3],
                    pose[3:],
                    twist[:3],
                    twist[3:],
                    wrench[:3],
                    wrench[3:],
                )
                pubs_markers[i].publish(markers)

        for pub in pubs_pose + pubs_twist + pubs_forces + pubs_wrench:
            pub.destroy()
        node.destroy_node()

    def close(self):
        """TODO."""
        self.logger.info(f"Estimator {self.drone_names} shutdown")
        self._shutdown.set()
        self._sub_process.join()
        self._pub_process.join()

        if self.settings.save_data:
            self.logger.info("Saving data...")
            for i, drone_name in enumerate(self.drone_names):
                filename = f"data_{drone_name}_"
                info = f"{self.settings.estimator_type}"
                if self.settings.estimator_type != "legacy":
                    info = info + f"_{self.settings.dynamics_model}"
                with open(filename + info + ".pkl", "wb") as f:
                    pickle.dump(self.data_est[i], f)
                with open(filename + "measurement" + ".pkl", "wb") as f:
                    pickle.dump(self.data_meas[i], f)


def launch_estimators(estimators: dict):
    """TODO."""
    processes = []
    seen_drone_names = []
    ctx = mp.get_context("spawn")
    shutdown = ctx.Event()

    # Estimators with identical settings (except for the drone name) are run as one batch
    batches = {}
    for k, settings in estimators.items():
        name = settings.drone_name
        if name in seen_drone_names:
            print(f"[ESTIMATOR_{name}]  Estimator for {name} already existing. Check settings")
            continue
        seen_drone_names.append(name)
        batch_settings = {k: v for k, v in settings.items() if k != "drone_name"}
        key = repr(sorted(batch_settings.items()))
        batches.setdefault(key, munchify(batch_settings | {"drone_names": []}))
        batches[key].drone_names.append(name)

    try:
        for settings in batches.values():
            # not sure if daemon should be True or False
            p = ctx.Process(target=launch_node, args=(settings, shutdown))
            processes.append(p)
            p.start()

        try:
            while True:
                time.sleep(10.0)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Terminating nodes...")

    finally:
        shutdown.set()

        for p in processes:
            p.join(timeout=2)

        for p in processes:
            if p.is_alive():
                print(f"Force terminating process {p.pid}")
                p.terminate()
                p.join()

        if rclpy.ok():
            rclpy.shutdown()
        print("All nodes terminated.")


def launch_node(settings: Munch, stop_event: Event):
    """TODO."""
    rclpy.init()
    estimator = MPEstimator(settings)
    try:
        estimator.run()
    finally:
        estimator.close()


if __name__ == "__main__":
    np.set_printoptions(linewidth=400, precision=3)  # TODO remove
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default=None, help="Path to settings file in CWD")
    parser.add_argument("--drone_name", default=None, help="Overwrite drone_name in estimator1")
    parser.add_argument(
        "--legacy", action="store_true", help="Overwrite estimator type in estimator1"
    )
    args = parser.parse_args()

    if args.settings is None:
        path = Path(__file__).parents[1] / "ros_nodes/estimators.toml"
    else:
        path = args.settings
    with open(path, "r") as f:
        estimators = munchify(toml.load(f))
    if args.drone_name is not None:
        estimators.estimator1.drone_name = args.drone_name  # overwrite drone_name
    if args.legacy:
        estimators.estimator1.estimator_type = "legacy"  # overwrite estimator type

    # Add debug to each estimator (if not already in place)
    for key, val in estimators.items():
        if not key.startswith("estimator"):
            continue
        for global_key, global_val in estimators.get("global", {}).items():
            if global_key not in val:
                val[global_key] = global_val

    estimators = munchify({k: v for k, v in estimators.items() if k.startswith("estimator")})

    launch_estimators(estimators)

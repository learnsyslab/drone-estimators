$$
\huge
\begin{aligned}
\hat{\boldsymbol{\mu}}_{x,k+1} &= \check{\boldsymbol{\mu}}_{x,k+1} + \mathbf{K}_{k+1} \mathbf{z}_{k+1} \\
\hat{\boldsymbol{\Sigma}}_{xx,k+1} &= ( \mathbf{I} - \mathbf{K}_{k+1} \mathbf{C}_{k+1} ) \check{\boldsymbol{\Sigma}}_{xx,k+1}
\end{aligned}
$$

---

Drone state estimators @ LSY. Contains model free (smoothing) and model based (EKF, UKF) state estimators for drones.

[![Python Version]][Python Version URL] [![PyPI]][PyPI URL] [![ROS 2]][ROS 2 URL] [![Ruff Check]][Ruff Check URL] [![Tests]][Tests URL] [![Docs]][Docs URL]

[Python Version]: https://img.shields.io/badge/python-3.11+-blue.svg
[Python Version URL]: https://www.python.org

[PyPI]: https://img.shields.io/pypi/v/drone-estimators.svg
[PyPI URL]: https://pypi.org/project/drone-estimators/

[ROS 2]: https://img.shields.io/badge/ROS%202-Kilted-22314E?logo=ros
[ROS 2 URL]: https://docs.ros.org/en/kilted/

[Ruff Check]: https://github.com/learnsyslab/drone-estimators/actions/workflows/ruff.yml/badge.svg?style=flat-square
[Ruff Check URL]: https://github.com/learnsyslab/drone-estimators/actions/workflows/ruff.yml

[Tests]: https://github.com/learnsyslab/drone-estimators/actions/workflows/testing.yml/badge.svg
[Tests URL]: https://github.com/learnsyslab/drone-estimators/actions/workflows/testing.yml

[Docs]: https://github.com/learnsyslab/drone-estimators/actions/workflows/docs.yml/badge.svg
[Docs URL]: https://learnsyslab.github.io/drone-estimators/

## Overview

The estimators are written against the [Python Array API standard](https://data-apis.org/array-api/latest/), so they run with NumPy as well as JAX. With JAX, they can be batched over many drones and jit compiled.

The [documentation](https://learnsyslab.github.io/drone-estimators/) covers installation, usage, the ROS 2 node, and the API reference.

## Installation

```bash
pip install drone-estimators
```

The ROS 2 node additionally needs a ROS 2 installation. The development setup below provides ROS 2 Kilted through [pixi](https://pixi.sh).

### Development

```bash
git clone https://github.com/learnsyslab/drone-estimators.git
cd drone-estimators
pixi shell
```

The default environment contains ROS 2 Kilted and installs `drone_estimators` in editable mode. On first activation, it also clones and builds [motion_capture_tracking](https://github.com/learnsyslab/motion_capture_tracking) in `ros_ws/`.

Run the tests with:

```bash
pixi run -e tests tests
```

## Usage

### Estimators

```python
import numpy as np

from drone_estimators.estimator_kalman import KalmanFilter

kf = KalmanFilter(dt=1 / 200, model="so_rpy", config="cf2x_L250")
kf.set_state(pos=np.zeros(3), quat=np.array([0.0, 0.0, 0.0, 1.0]))

# Command as roll, pitch, yaw [rad] and collective thrust [N]
kf.predict(dt=1 / 200, command=np.array([0.0, 0.0, 0.0, 0.3]))
estimate = kf.correct(pos=np.array([0.0, 0.0, 0.001]), quat=np.array([0.0, 0.0, 0.0, 1.0]))
print(estimate.pos, estimate.vel, estimate.ang_vel)
```

To estimate many drones at once, pass a `batch_shape` and enable jit compilation:

```python
import jax

jax.config.update("jax_enable_x64", True)  # The jitted filters need 64 bit precision

kf = KalmanFilter(dt=1 / 200, model="so_rpy", config="cf2x_L250", batch_shape=(10,), jit=True)
```

The differentiation-based filter is `DiffFilter` in `drone_estimators.estimator_diff`.

### ROS 2 node

```bash
python -m drone_estimators.ros_nodes.ros2_node --drone_name cf1 --estimator_type ukf
python -m drone_estimators.ros_nodes.ros2_node --drone_name cf1 --estimator_type diff
python -m drone_estimators.ros_nodes.ros2_node --settings path/to/estimators.toml
```

The settings are read from [`estimators.toml`](drone_estimators/ros_nodes/estimators.toml). To estimate several drones, add one `[estimatorX]` section per drone and pass your file with `--settings`. Drones with identical settings are batched into one estimator.

The node subscribes to `/tf` (e.g., from [motion_capture_tracking](https://github.com/learnsyslab/motion_capture_tracking)) and to `/drones/<name>/command` (roll, pitch, yaw in deg and thrust in PWM, needed by the UKF). It publishes the estimates to `/drones/<name>/estimate/{pose,twist,wrench,rotor_vel}`. The node only communicates with nodes on the same machine (`ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`).

To read the estimates in Python without blocking, use the `ROSConnector`:

```python
import rclpy

from drone_estimators.ros_nodes.ros2_connector import ROSConnector

rclpy.init()
conn = ROSConnector(estimator_names=["cf1"], timeout=10.0)
print(conn.pos["cf1"], conn.vel["cf1"])
conn.close()
```

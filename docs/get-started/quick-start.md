# Quick Start

This page shows how to use the estimators directly from Python. To run them on a real system with motion capture data, see [ROS 2](ros2.md).

All estimators share the same interface: `set_state` initializes the estimate with a first measurement, `predict` advances the estimate in time, and `correct` fuses a new pose measurement. Both `predict` and `correct` return the current estimate.

| Variable | Description | Shape | Convention |
|---|---|---|---|
| `pos` | Measured position in world frame | `(3,)` | m |
| `quat` | Measured orientation as unit quaternion | `(4,)` | scalar-last `xyzw` |
| `dt` | Time since the last prediction | `()` | s |

## Unscented Kalman Filter

The `KalmanFilter` uses a dynamics model from [crazyflow](https://github.com/learnsyslab/crazyflow) to predict the state between measurements. Since the dynamics depend on what the drone is commanded to do, `predict` takes the latest command. The command has to match the input of the model, e.g., roll, pitch, yaw (rad) and collective thrust (N) for the `so_rpy` models, or the four motor velocities (RPM) for `first_principles`.

```python
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)  # The jitted estimators need 64 bit precision

from drone_estimators.estimator_kalman import KalmanFilter

kf = KalmanFilter(
    dt=1 / 200,  # Nominal time step [s]
    model="so_rpy_rotor_drag",  # Any crazyflow dynamics model
    config="cf21B_500",  # Any drone the model has parameters for
    estimate_rotor_vel=True,
    estimate_dist_f=True,
    jit=True,
)
kf.set_state(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0, 1.0]))  # pos, quat (xyzw)

cmd = np.array([0.0, 0.0, 0.0, 0.42])  # roll, pitch, yaw [rad], collective thrust [N]
for k in range(200):
    pos = np.array([0.0, 0.0, 1.0 + 0.1 * k / 200])  # e.g., from a motion capture system
    quat = np.array([0.0, 0.0, 0.0, 1.0])
    kf.predict(1 / 200, cmd)
    estimate = kf.correct(pos, quat)

print(estimate.pos, estimate.vel, estimate.dist_f)
```

The estimate is an `EstimatorData` with the fields `pos`, `quat`, `vel`, `ang_vel` (body frame), and, if enabled, `rotor_vel`, `dist_f`, and `dist_t`. Fields that are not estimated are `None`.

!!! note "Rotor velocities"
    `estimate_rotor_vel=True` requires a model with rotor dynamics, i.e., `first_principles`, `so_rpy_rotor`, or `so_rpy_rotor_drag`.

## Legacy estimator

The `StateEstimator` needs neither a model nor commands. It low pass filters the measurements and their numerical derivatives with five time constants for the position, velocity, acceleration, orientation, and angular velocity.

```python
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from drone_estimators.estimator_legacy import StateEstimator

estimator = StateEstimator((0.0001, 0.007, 0.09, 0.005, 0.07))  # Filter time constants
estimator.set_state(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
for k in range(1, 200):
    pos = np.array([0.1 * k / 200, 0.0, 0.0])
    quat = np.array([0.0, 0.0, 0.0, 1.0])
    estimator.predict(1 / 200)
    estimate = estimator.correct(pos, quat)

print(estimate.pos, estimate.vel, estimate.ang_vel)
```

## JIT compilation

With `jit=True`, the filter functions are compiled with `jax.jit` on construction, so the first measurement is not delayed by the compilation. The UKF defaults to `jit=False`, the legacy estimator to `jit=True`.

!!! warning "64 bit precision"
    The jitted estimators need 64 bit precision. The covariances of the UKF span many orders of magnitude, and the numerical derivatives of the legacy estimator amplify rounding errors. Call `jax.config.update("jax_enable_x64", True)` before creating a jitted estimator, otherwise it raises a `RuntimeError`.

## Batching

All estimators support leading batch dimensions, e.g., one estimator for a whole swarm. Pass the `batch_shape` on construction and arrays with a leading batch dimension afterwards. Drones without a new measurement, e.g., because they are occluded, are excluded from the correction with a `mask`. The time step of `predict` can also differ per drone.

```python
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from drone_estimators.estimator_kalman import KalmanFilter

n_drones = 3
kf = KalmanFilter(1 / 200, "so_rpy", "cf2x_L250", batch_shape=(n_drones,), jit=True)
kf.set_state(np.zeros((n_drones, 3)), np.tile([0.0, 0.0, 0.0, 1.0], (n_drones, 1)))

pos = np.zeros((n_drones, 3))
quat = np.tile([0.0, 0.0, 0.0, 1.0], (n_drones, 1))
has_measurement = np.array([True, False, True])  # e.g., the second drone is occluded
kf.predict(1 / 200)
estimate = kf.correct(pos, quat, mask=has_measurement)
print(estimate.pos.shape)  # (3, 3)
```

$$
\begin{aligned}
\hat{\boldsymbol{\mu}}_{x,k+1} &= \check{\boldsymbol{\mu}}_{x,k+1} + \mathbf{K}_{k+1} \mathbf{z}_{k+1} \\
\hat{\boldsymbol{\Sigma}}_{xx,k+1} &= ( \mathbf{I} - \mathbf{K}_{k+1} \mathbf{C}_{k+1} ) \check{\boldsymbol{\Sigma}}_{xx,k+1}
\end{aligned}
$$

Welcome to **Drone Estimators** - a Python package providing state estimators for quadrotor drones. The estimators turn noisy pose measurements, e.g., from a motion capture system, into estimates of the full state: position, orientation, linear and angular velocity, and optionally rotor velocities and disturbance forces and torques.

```python
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)  # The jitted estimators need 64 bit precision

from drone_estimators.estimator_kalman import KalmanFilter

kf = KalmanFilter(1 / 200, model="so_rpy", config="cf2x_L250", jit=True)
kf.set_state(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))  # pos, quat (xyzw)

kf.predict(1 / 200, np.array([0.0, 0.0, 0.0, 0.31]))  # roll, pitch, yaw [rad], thrust [N]
estimate = kf.correct(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))  # Measured pos, quat
```

## Estimators

| Estimator | Class | Description |
|---|---|---|
| Differentiation-based filter | `DiffFilter` | Model free smoothing estimator. Low pass filters the measurements and their numerical derivatives. Needs no model and no inputs. |
| EKF | `KalmanFilter(filter_type="EKF")` | Extended Kalman Filter with the dynamics models of [crazyflow](https://github.com/learnsyslab/crazyflow). Uses the commands sent to the drone and can additionally estimate rotor velocities and disturbance forces and torques. Propagates the covariance with the Jacobian of the dynamics at hover, computed with JAX. |
| UKF | `KalmanFilter` | Unscented Kalman Filter with the same models and states as the EKF. Propagates the covariance with sigma points through the nonlinear dynamics. |

Both Kalman filters support every dynamics model of crazyflow (`first_principles`, `so_rpy`, `so_rpy_rotor`, `so_rpy_rotor_drag`) with every drone configuration the model has parameters for. For `first_principles`, see the warning in the [Quick Start](get-started/quick-start.md#unscented-kalman-filter).

## Array API, batching, and JIT

The estimators are written against the [Array API standard](https://data-apis.org/array-api/latest/) and work with NumPy and JAX arrays. They are pure functions under the hood, which means they can be compiled with `jax.jit` for fast estimation loops. All estimators support leading batch dimensions, so a single estimator can track a whole swarm of drones at once.

## ROS 2

The package ships a ROS 2 node that runs the estimators on motion capture data published on `/tf` and publishes the estimates for each drone. The `ROSConnector` gives non-blocking access to these estimates from your own Python code. See [ROS 2](get-started/ros2.md).

## Next steps

- New here? Start with [Installation](get-started/installation.md) and [Quick Start](get-started/quick-start.md).
- Running on a real system? See [ROS 2](get-started/ros2.md).
- Looking for the details? See the [API Reference](reference/index.md).

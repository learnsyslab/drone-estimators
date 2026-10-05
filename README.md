$$
\huge \displaystyle \hat{\boldsymbol{\mu}}_{x,k+1} = \check{\boldsymbol{\mu}}_{x,k+1} + \mathbf{K}_{k+1} \mathbf{z}_{k+1} \\
\hat{\boldsymbol{\Sigma}}_{xx, k+1} = ( \mathbf{I}-\mathbf{K}_{k+1} \mathbf{C}_k+1) \check{\boldsymbol{\Sigma}}_{xx, k+1}
$$

---

Drone state estimators @ LSY. Contains model free (smoothing) and model based (EKF, UKF) state estimators for drones.

[![Python Version]][Python Version URL] [![Ruff Check]][Ruff Check URL] [![Tests]][Tests URL]

[Python Version]: https://img.shields.io/badge/python-3.11+-blue.svg
[Python Version URL]: https://www.python.org

[Ruff Check]: https://github.com/utiasDSL/drone-estimators/actions/workflows/ruff.yml/badge.svg?style=flat-square
[Ruff Check URL]: https://github.com/utiasDSL/drone-estimators/actions/workflows/ruff.yml

[Tests]: https://github.com/utiasDSL/drone-estimators/actions/workflows/testing.yml/badge.svg
[Tests URL]: https://github.com/utiasDSL/drone-estimators/actions/workflows/testing.yml

## Installation
### Normal 
Simply install with pip
```bash
pip install drone-estimators
```

### Development
Clone repository and install repository:

```bash
git clone git@github.com:utiasDSL/drone-estimators.git
cd drone-estimators
pixi install
```

Now activate the `jazzy` environment and install the package in editable mode
```bash
pixi shell -e jazzy
uv pip install -e .
```

If you want to have `crazyflow` (which provides the dynamics models) in editable mode, simply install it too with pip, i.e.,
```bash
cd ../crazyflow
uv pip install -e .
```

## Usage
Either use the estimators directly:

```bash
from drone_estimators.estimator_kalman import KalmanFilter
```

or run the `ROS2` node with:
```bash
python -m drone_estimators.ros_nodes.ros2_node
```

For the latter, you can either add your specific drone to be estimated with default settings or add all drones to the `estimators.toml` file (editable mode), which can be passed as an argument
```bash
python -m drone_estimators.ros_nodes.ros2_node --drone <your_drone>
python -m drone_estimators.ros_nodes.ros2_node --settings <path/to/your/estimators.toml>
```
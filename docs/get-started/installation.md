# Installation

## Requirements

Python 3.11 or later. The dynamics models of the UKF come from [crazyflow](https://github.com/learnsyslab/crazyflow), which is installed as a dependency. The ROS 2 node additionally needs a ROS 2 installation, see [Developer install with pixi](#developer-install-with-pixi).

## Install from PyPI

```bash
pip install drone-estimators
```

## Developer install with pixi

For development and for running the ROS 2 node, we use [pixi](https://pixi.sh). Pixi locks the full dependency tree, including ROS 2 Kilted from [RoboStack](https://robostack.github.io/), into reproducible environments.

```bash
git clone https://github.com/learnsyslab/drone-estimators.git
cd drone-estimators
pixi install
```

The available environments are:

| Environment | Activate with | Includes |
|---|---|---|
| `default` | `pixi shell` | Package (editable) + ROS 2 Kilted |
| `tests` | `pixi shell -e tests` | Default + pytest |
| `docs` | `pixi shell -e docs` | Package (editable) + ProperDocs for building the documentation |
| `dist` | `pixi shell -e dist` | Default + build tools for releasing the package |

The ROS 2 environments clone and build [motion_capture_tracking](https://github.com/learnsyslab/motion_capture_tracking) into `ros_ws` on activation.

Each environment has pixi tasks associated with it:

```bash
# Run the test suite
pixi run -e tests tests

# Serve the documentation locally (live reload)
pixi run -e docs docs-serve

# Build the documentation
pixi run -e docs docs-build
```

If you want to have `crazyflow` in editable mode as well, install it into the environment from a local clone:

```bash
pixi shell
cd ../crazyflow
uv pip install -e .
```

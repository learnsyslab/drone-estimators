"""Data shared by all estimators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from flax.struct import dataclass

if TYPE_CHECKING:
    from drone_estimators._typing import Array  # To be changed to array_api_typing later


@dataclass
class EstimatorData:
    """TODO."""

    pos: Array
    quat: Array
    vel: Array
    ang_vel: Array
    rotor_vel: Array | None
    dist_f: Array | None
    dist_t: Array | None
    covariance: Array  # Covariance matrix

    sigmas_f: Array
    sigmas_h: Array

    u: Array  # input
    z: Array  # measurement
    dt: float

    @classmethod
    def create_empty(
        cls,
        rotor_vel: bool = False,
        dist_f: bool = False,
        dist_t: bool = False,
        dim_u: int = 4,
        dim_z: int = 7,
    ) -> EstimatorData:
        """TODO."""
        pos = np.zeros(3)
        quat = np.array([0, 0, 0, 1])
        vel = np.zeros(3)
        ang_vel = np.zeros(3)
        dim_x = 13
        if rotor_vel:
            rotor_vel = np.zeros(4)
            dim_x = dim_x + 4
        else:
            rotor_vel = None
        if dist_f:
            dist_f = np.zeros(3)
            dim_x = dim_x + 3
        else:
            dist_f = None
        if dist_t:
            dist_t = np.zeros(3)
            dim_x = dim_x + 3
        else:
            dist_t = None

        covariance = np.eye(dim_x)

        sigmas_f = np.zeros((2 * dim_x + 1, dim_x))
        sigmas_h = np.zeros((2 * dim_x + 1, dim_z))

        u = np.zeros(dim_u)  # input
        z = np.zeros(dim_z)  # measurement
        dt = 1

        return cls(
            pos,
            quat,
            vel,
            ang_vel,
            rotor_vel,
            dist_f,
            dist_t,
            covariance,
            sigmas_f,
            sigmas_h,
            u,
            z,
            dt,
        )

    @classmethod
    def create(
        cls,
        pos: Array,
        quat: Array,
        vel: Array,
        ang_vel: Array,
        rotor_vel: Array | None = None,
        dist_f: Array | None = None,
        dist_t: Array | None = None,
    ) -> EstimatorData:
        """TODO."""
        dim_x = 13
        if rotor_vel is not None:
            dim_x = dim_x + 4
        if dist_f is not None:
            dim_x = dim_x + 3
        if dist_t is not None:
            dim_x = dim_x + 3

        covariance = np.eye(dim_x)

        sigmas_f = np.zeros((2 * dim_x + 1, dim_x))
        sigmas_h = np.zeros((2 * dim_x + 1, 7))

        u = np.zeros(4)  # input
        z = np.zeros(7)  # measurement
        dt = 1

        return cls(
            pos,
            quat,
            vel,
            ang_vel,
            rotor_vel,
            dist_f,
            dist_t,
            covariance,
            sigmas_f,
            sigmas_h,
            u,
            z,
            dt,
        )

    @classmethod
    def as_state_array(cls, data: EstimatorData) -> Array:
        """Returns the state as an array."""
        xp = data.pos.__array_namespace__()
        x = xp.concat((data.pos, data.quat, data.vel, data.ang_vel), axis=-1)
        if data.rotor_vel is not None:
            x = xp.concat((x, data.rotor_vel), axis=-1)
        if data.dist_f is not None:
            x = xp.concat((x, data.dist_f), axis=-1)
        if data.dist_t is not None:
            x = xp.concat((x, data.dist_t), axis=-1)
        return x

    @classmethod
    def from_state_array(cls, data: EstimatorData, array: Array) -> EstimatorData:
        """Updates data in the given structure based on a given array."""
        pos = array[..., 0:3]
        quat = array[..., 3:7]
        vel = array[..., 7:10]
        ang_vel = array[..., 10:13]
        idx = 13
        if data.rotor_vel is not None:
            rotor_vel = array[..., idx : idx + 4]
            idx = idx + 4
        else:
            rotor_vel = None
        if data.dist_f is not None:
            dist_f = array[..., idx : idx + 3]
            idx = idx + 3
        else:
            dist_f = None
        if data.dist_t is not None:
            dist_t = array[..., idx : idx + 3]
            idx = idx + 3
        else:
            dist_t = None

        return data.replace(
            pos=pos,
            quat=quat,
            vel=vel,
            ang_vel=ang_vel,
            rotor_vel=rotor_vel,
            dist_f=dist_f,
            dist_t=dist_t,
        )

    @classmethod
    def get_state_dim(cls, data: EstimatorData) -> int:
        """Returns the dimension of the state."""
        dim_x = 13
        if data.rotor_vel is not None:
            dim_x = dim_x + 4
        if data.dist_f is not None:
            dim_x = dim_x + 3
        if data.dist_t is not None:
            dim_x = dim_x + 3
        return dim_x

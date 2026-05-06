"""
Class for rod rotation task.
"""

from typing import List

import jax
import jax.numpy as np
import numpy as onp

from swarmrl.components.colloid import Colloid
from swarmrl.tasks.task import Task
from swarmrl.utils.colloid_utils import compute_normalized_rod_torques


class RodTorque(Task):
    """
    Reward colloids for applying torque that rotates a rod in the commanded direction.
    """

    def __init__(
        self,
        rod_type: int = 1,
        particle_type: int = 0,
        direction: str = "CCW",
        angular_velocity_scale: int = 100,
        velocity_history_size: int = 100,
    ):
        """
        Constructor for the Rot-Torque task.

        Parameters
        ----------
        rod_type : int (default=1)
                Type of particle making up the rod.
        particle_type : int (default=0)
                Type of particle receiving the reward.
        direction : str (default="CCW")
                Direction of the rod to rotate.
        angular_velocity_scale : float (default=100.0)
                The amount the velocity is scaled by to get the reward.
        velocity_history_size : int (default=100)
                Number of steps to average the velocity over.
        """
        super().__init__(particle_type=particle_type)
        self.rod_type = rod_type

        if velocity_history_size < 1:
            raise ValueError("Velocity history must be greater than 0.")
        else:
            self.velocity_history_size = velocity_history_size

        if angular_velocity_scale < 1:
            raise ValueError(
                "Angular velocity scale must be greater than 0. For rotational"
                " direction, use 'CW' or 'CCW'."
            )

        if direction == "CW":
            angular_velocity_scale *= -1  # CW is negative

        self.angular_velocity_scale = angular_velocity_scale
        self._velocity_history_list = np.full(velocity_history_size, np.nan)

        self.decomp_fn = jax.jit(compute_normalized_rod_torques)

    def initialize(self, colloids: List[Colloid]):
        """
        Prepare the task for running.

        In this case, as all rod directors are the same, we
        only need to take one for the historical value.

        Parameters
        ----------
        colloids : List[Colloid]
                List of colloids to be used in the task.

        Returns
        -------
        Updates the class state.
        """
        self._velocity_history_list = np.full(self.velocity_history_size, np.nan)
        for item in colloids:
            if item.type == self.rod_type:
                self._historic_rod_director = onp.copy(item.director)
                break
        else:
            raise ValueError(
                f"RodTorque.initialize: no rod particles found (type {self.rod_type})."
            )

    def _compute_signed_rod_torques(
        self,
        rod_positions: np.ndarray,
        colloid_directors: np.ndarray,
        colloid_positions: np.ndarray,
    ) -> np.ndarray:
        """
        Compute signed z-torque contributions on the rod.

        Parameters
        ----------
        rod_positions : np.ndarray (n_rod, 3)
                Positions of the rod particles.
        colloid_directors : np.ndarray (n_colloids, 3)
                Directors of the colloids.
        colloid_positions : np.ndarray (n_colloids, 3)
                Positions of the colloids.

        Returns
        -------
        torques : np.ndarray (n_colloids, )
                Signed z-torque contributions on the rod for each colloid.
        """
        torques = self.decomp_fn(rod_positions, colloid_directors, colloid_positions)[
            :, 2
        ]
        return torques

    def _compute_angular_velocity(self, new_director: np.ndarray):
        """
        Compute the average angular velocity of the rod.

        Parameters
        ----------
        new_director : np.ndarray (3, )
                New rod director.

        Returns
        -------
        angular_velocity : float
                Angular velocity of the rod
        """
        angular_velocity = np.arctan2(
            np.cross(self._historic_rod_director[:2], new_director[:2]),
            np.dot(self._historic_rod_director[:2], new_director[:2]),
        )

        # Convert to degree for easier handling
        angular_velocity = np.rad2deg(angular_velocity)

        # Update the historical rod director and velocity.
        self._historic_rod_director = new_director
        self._velocity_history_list = np.roll(self._velocity_history_list, -1)
        self._velocity_history_list = self._velocity_history_list.at[-1].set(
            angular_velocity
        )

        # Return the average over observed values only.
        return np.nanmean(self._velocity_history_list)

    def _filter_torques_by_direction(
        self,
        colloid_torques_on_rod: np.ndarray,
    ) -> np.ndarray:
        """
        Keep only torque magnitudes in the commanded rotation direction.
        Relevant for reward computation.

        Parameters
        ----------
        colloid_torques_on_rod : np.ndarray (n_colloids, )
                Torques of the colloids on the rod.

        Returns
        -------
        torques_in_direction : np.ndarray (n_colloids, )
                Positive torque magnitudes in the commanded direction.
        """
        # CCW
        if self.angular_velocity_scale > 0.0:
            torques_in_direction = np.where(
                colloid_torques_on_rod < 0.0, -colloid_torques_on_rod, 0.0
            )
        # CW
        else:
            torques_in_direction = np.where(
                colloid_torques_on_rod > 0.0, colloid_torques_on_rod, 0.0
            )

        return torques_in_direction

    def _compute_directed_torque_reward(
        self,
        rod_directors: np.ndarray,
        rod_positions: np.ndarray,
        colloid_directors: np.ndarray,
        colloid_positions: np.ndarray,
    ):
        """
        Compute reward from commanded-direction torque and rod rotation.

        Parameters
        ----------
        rod_directors : np.ndarray (n_rod, 3)
                Directors of the rod particles.
        rod_positions : np.ndarray (n_rod, 3)
                Positions of the rod particles.
        colloid_directors : np.ndarray (n_colloids, 3)
                Directors of the colloids.
        colloid_positions : np.ndarray (n_colloids, 3)
                Positions of the colloids.

        Returns
        -------
        rewards : np.ndarray (n_colloids, )
                Rewards for each colloid.
        """

        colloid_torques_on_rod = self._compute_signed_rod_torques(
            rod_positions, colloid_directors, colloid_positions
        )
        filtered_torques = self._filter_torques_by_direction(colloid_torques_on_rod)
        angular_velocity = self._compute_angular_velocity(rod_directors[0])
        if self.angular_velocity_scale > 0.0:
            clipped_velocity = np.clip(angular_velocity, 0.0, None)
        else:
            clipped_velocity = np.clip(angular_velocity, None, 0.0)

        return (
            filtered_torques
            * np.abs(clipped_velocity)
            * np.abs(self.angular_velocity_scale)
        )

    def __call__(self, colloids: List[Colloid]):
        """
        Compute the reward.

        Parameters
        ----------
        colloids : List[Colloid] (n_colloids, )
                List of colloids to be used in the task.

        Returns
        -------
        rewards : List[float] (n_colloids, )
                Rewards for each colloid.
        """
        # Collect all rod particles
        rod = [colloid for colloid in colloids if colloid.type == self.rod_type]
        rod_positions = np.array([colloid.pos for colloid in rod])
        rod_directors = np.array([colloid.director for colloid in rod])

        # Collect all colloids
        chosen_colloids = [
            colloid for colloid in colloids if colloid.type == self.particle_type
        ]
        colloid_positions = np.array([colloid.pos for colloid in chosen_colloids])
        colloid_directors = np.array([colloid.director for colloid in chosen_colloids])

        rewards = self._compute_directed_torque_reward(
            rod_directors, rod_positions, colloid_directors, colloid_positions
        )

        return rewards

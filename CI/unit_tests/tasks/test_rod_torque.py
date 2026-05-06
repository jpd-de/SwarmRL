"""
Unit test for the rod torque task.
"""

import numpy as np
import pytest

from swarmrl.components import Colloid
from swarmrl.tasks.object_movement.rod_torque import RodTorque


def create_trajectory(direction_scale: int = 1):
    """
    Create a trajectory for the tests.
    """
    colloids = []
    angle = 0.0

    starting_director = np.array([1, 0, 0])
    for _ in range(100):
        angle += np.deg2rad(direction_scale * 45)
        director = np.array([np.cos(angle), np.sin(angle), 0])
        colloids.append(
            Colloid(pos=np.array([0, 0, 0]), id=0, director=director, type=1)
        )

    return starting_director, colloids


class TestRodTorque:
    """
    Test suite for the rod rotations.
    """

    @classmethod
    def setup_class(cls):
        """
        Setup the test class.
        """
        cls.reference_velocity = 45.0

    def test_ccw_rotation(self):
        """
        Setup the test class.
        """
        task = RodTorque(direction="CCW", angular_velocity_scale=1.0)

        # Test positive rewards.
        starting_director, colloids = create_trajectory(direction_scale=1)
        task._historic_rod_director = starting_director

        # Test that the velocity is correct
        for colloid in colloids:
            velocity = task._compute_angular_velocity(colloid.director)

            assert velocity == pytest.approx(self.reference_velocity)

        # Test opposite rotation direction.
        task = RodTorque(direction="CCW", angular_velocity_scale=1.0)
        starting_director, colloids = create_trajectory(direction_scale=-1)

        task._historic_rod_director = starting_director

        # Test that the velocity is correct
        for colloid in colloids:
            velocity = task._compute_angular_velocity(colloid.director)

            assert velocity == pytest.approx(-self.reference_velocity)

    def test_cw_rotation(self):
        """
        Setup the test class.
        """
        task = RodTorque(
            direction="CW", angular_velocity_scale=1.0, velocity_history_size=1
        )

        # Test positive rewards.
        starting_director, colloids = create_trajectory(direction_scale=-1)
        task._historic_rod_director = starting_director

        # Test that the velocity is correct
        for colloid in colloids:
            velocity = task._compute_angular_velocity(colloid.director)

            assert velocity == pytest.approx(self.reference_velocity * -1)

        # Test opposite rotation direction.
        task = RodTorque(
            direction="CW", angular_velocity_scale=1.0, velocity_history_size=1
        )
        starting_director, colloids = create_trajectory(direction_scale=1)
        task._historic_rod_director = starting_director

        # Test that the velocity is correct
        for colloid in colloids:
            velocity = task._compute_angular_velocity(colloid.director)
            assert velocity == pytest.approx(self.reference_velocity)

    def test_velocity_history(self):
        """
        Setup the test class.
        """
        task = RodTorque(
            direction="CCW", angular_velocity_scale=1.0, velocity_history_size=100
        )
        velocity_history = np.full(100, np.nan)

        # Test positive rewards.
        starting_director, colloids = create_trajectory(direction_scale=1)
        task._historic_rod_director = starting_director

        # Test that the velocity is correct
        for index, colloid in enumerate(colloids):
            velocity = task._compute_angular_velocity(colloid.director)
            velocity_history[index] = self.reference_velocity
            assert velocity == pytest.approx(np.nanmean(velocity_history))

    def test_compute_signed_rod_torques(self):
        """
        Setup the test class.
        """
        task = RodTorque(
            direction="CCW", angular_velocity_scale=1.0, velocity_history_size=1
        )

        torque = task._compute_signed_rod_torques(
            rod_positions=np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]),
            colloid_directors=np.array([[0.0, 1.0, 0.0]]),
            colloid_positions=np.array([[1.0, -1.0, 0.0]]),
        )
        assert torque == pytest.approx([-1])
        torque = task._compute_signed_rod_torques(
            rod_positions=np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]),
            colloid_directors=np.array([[0.0, -1.0, 0.0]]),
            colloid_positions=np.array([[-1.0, 1.0, 0.0]]),
        )
        assert torque == pytest.approx([-1])
        torque = task._compute_signed_rod_torques(
            rod_positions=np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]),
            colloid_directors=np.array([[1.0, 0.0, 0.0]]),
            colloid_positions=np.array([[100.0, 100.0, 0.0]]),
        )
        assert torque == pytest.approx([0])
        torque = task._compute_signed_rod_torques(
            rod_positions=np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]),
            colloid_directors=np.array([[0.0, 1.0, 0.0]]),
            colloid_positions=np.array([[-1.0, -1.0, 0.0]]),
        )
        assert torque == pytest.approx([1])

    def test_reward_keeps_only_commanded_direction_torques(self):
        """
        Reward should only keep torque contributions in the commanded direction.
        """
        rod_positions = np.array([[0.0, 0.0, 0.0]])
        rod_directors = np.array([[np.sqrt(0.5), np.sqrt(0.5), 0.0]])
        colloid_positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        colloid_directors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        ccw_task = RodTorque(
            direction="CCW", angular_velocity_scale=1.0, velocity_history_size=1
        )
        ccw_task._historic_rod_director = np.array([1.0, 0.0, 0.0])
        ccw_task._compute_signed_rod_torques = lambda *_: np.array([-2.0, 3.0])

        ccw_rewards = ccw_task._compute_directed_torque_reward(
            rod_directors, rod_positions, colloid_directors, colloid_positions
        )

        assert ccw_rewards == pytest.approx([90.0, 0.0])

        cw_task = RodTorque(
            direction="CW", angular_velocity_scale=1.0, velocity_history_size=1
        )
        cw_task._historic_rod_director = np.array([1.0, 0.0, 0.0])
        cw_task._compute_signed_rod_torques = lambda *_: np.array([-2.0, 3.0])

        cw_rewards = cw_task._compute_directed_torque_reward(
            np.array([[np.sqrt(0.5), -np.sqrt(0.5), 0.0]]),
            rod_positions,
            colloid_directors,
            colloid_positions,
        )

        assert cw_rewards == pytest.approx([0.0, 135.0])

    def test_reward_clips_ccw_wrong_direction_velocity(self):
        """
        CCW rewards should be zero when the rod rotates in the wrong direction.
        """
        task = RodTorque(
            direction="CCW", angular_velocity_scale=1.0, velocity_history_size=1
        )
        task._historic_rod_director = np.array([1.0, 0.0, 0.0])
        task._compute_signed_rod_torques = lambda *_: np.array([-2.0, -3.0])

        rewards = task._compute_directed_torque_reward(
            np.array([[np.sqrt(0.5), -np.sqrt(0.5), 0.0]]),
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        )

        assert rewards == pytest.approx([0.0, 0.0])

    def test_reward_clips_cw_wrong_direction_velocity(self):
        """
        CW rewards should be zero when the rod rotates in the wrong direction.
        """
        task = RodTorque(
            direction="CW", angular_velocity_scale=1.0, velocity_history_size=1
        )
        task._historic_rod_director = np.array([1.0, 0.0, 0.0])
        task._compute_signed_rod_torques = lambda *_: np.array([2.0, 3.0])

        rewards = task._compute_directed_torque_reward(
            np.array([[np.sqrt(0.5), np.sqrt(0.5), 0.0]]),
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        )

        assert rewards == pytest.approx([0.0, 0.0])

"""
Test the connection between simulation and experiment.
To this end, set up a mock-connection that acts as the experiment does.
The assertions happen inside the mock-connections.

If the experiment changes (i.e. the matlab code), the changes need to be reflected in
MockConnection to ensure this test stays up-to-date.
"""

import enum
import struct
import unittest as ut

import numpy as np

import swarmrl.engine.real_experiment
from swarmrl.actions import Action
from swarmrl.agents.agent import Agent
from swarmrl.agents.dummy_models import ConstForce
from swarmrl.components import Colloid
from swarmrl.force_functions import ForceFunction


class MessageType(enum.Enum):
    DATA_SIZE = 0
    ACTUAL_DATA = 1


class MockConnection:
    """
    The experiment sends
    - 1st the data size of the particle property matrix
    - 2nd the actual matrix

    The experiment receives
    - a matrix of actions
    """

    def __init__(self, n_partcl: int, box_l: float):
        self.n_partcl = n_partcl
        self.box_l = box_l

        self.actual_data_size = self.n_partcl * 4  # [x y theta id]
        self.ids = np.array(range(self.n_partcl))
        self.sendall_calls = 0
        self.receive_batch_idx = 0

        # the first message of the experiment will always be the data size
        self.next_message = MessageType.DATA_SIZE

    def make_particle_properties(self):
        xs = self.box_l * np.random.random((self.n_partcl,))
        ys = self.box_l * np.random.random((self.n_partcl,))
        thetas = 2 * np.pi * np.random.random((self.n_partcl,))
        return np.column_stack((xs, ys, thetas, self.ids.astype(float)))

    def recv(self, data_size: int) -> bytes:
        """
        Supply the engine with data, alternating between the data size and a matrix of
        random particle properties.
        """
        if self.next_message == MessageType.DATA_SIZE:
            assert data_size == 8
            actual_data_size_bytes = struct.pack("I", self.actual_data_size)
            self.next_message = MessageType.ACTUAL_DATA
            return actual_data_size_bytes
        elif self.next_message == MessageType.ACTUAL_DATA:
            assert data_size == 8 * self.actual_data_size

            partcl_props = self.make_particle_properties()

            # experiment sends data C-style flattened
            partcl_props = partcl_props.flatten("C")
            partcl_props_bytes = struct.pack(
                str(len(partcl_props)) + "d", *partcl_props
            )

            self.next_message = MessageType.DATA_SIZE
            return partcl_props_bytes

    def sendall(self, data: bytes):
        """
        Receive data from the engine and check that the values are sensible
        """
        # experiment expects data to be F-Style flattened
        data_unpacked = np.array(struct.unpack(str(len(data) // 8) + "d", data))
        data_matrix = data_unpacked.reshape((-1, 2), order="F")
        shape = np.shape(data_matrix)
        assert shape == (self.n_partcl, 2)
        assert np.all(data_matrix[:, 0].astype(int) == self.ids)
        for action_id in data_matrix[:, 1]:
            assert (
                action_id in swarmrl.engine.real_experiment.experiment_actions.values()
            )
        self.sendall_calls += 1


class SequencedMockConnection(MockConnection):
    def __init__(self, batches):
        super().__init__(n_partcl=len(batches[0]), box_l=1.0)
        self._batches = batches

    def make_particle_properties(self):
        batch = self._batches[self.receive_batch_idx]
        self.receive_batch_idx += 1
        return batch


class TrackingAgent(Agent):
    def __init__(self, connection=None):
        self.connection = connection
        self.events = []
        self.reward_sendall_calls = []
        self.action_batches = []
        self.reward_batches = []
        self.action_colloids = None
        self.reward_colloids = None

    def _snapshot(self, colloids):
        return [tuple(colloid.pos.tolist()) for colloid in colloids]

    def calc_action(self, colloids):
        self.events.append("action")
        self.action_colloids = colloids
        self.action_batches.append(self._snapshot(colloids))
        return [Action() for _ in colloids]

    def calc_reward(self, colloids, external_reward=0.0):
        self.events.append("reward")
        self.reward_colloids = colloids
        self.reward_batches.append(self._snapshot(colloids))
        if self.connection is not None:
            self.reward_sendall_calls.append(self.connection.sendall_calls)


class TestRealExperiment(ut.TestCase):
    def test_communication(self):
        connection = MockConnection(n_partcl=17, box_l=8.765)
        runner = swarmrl.engine.real_experiment.RealExperiment(connection)
        runner.setup_simulation()
        agent = ConstForce(123)
        force_fn = ForceFunction({"0": agent})
        runner.integrate(10, force_fn)
        runner.finalize()

    def test_reward_collection_uses_subsequent_observation(self):
        first_batch = np.array([
            [10.0, 1.0, 0.0, 0.0],
            [11.0, 2.0, 0.5, 1.0],
            [12.0, 3.0, 1.0, 2.0],
        ])
        second_batch = np.array([
            [20.0, 4.0, 0.1, 0.0],
            [21.0, 5.0, 0.6, 1.0],
            [22.0, 6.0, 1.1, 2.0],
        ])
        connection = SequencedMockConnection([first_batch, second_batch])
        runner = swarmrl.engine.real_experiment.RealExperiment(connection)
        agent = TrackingAgent(connection=connection)
        force_fn = ForceFunction({"0": agent})

        runner.integrate(1, force_fn)

        self.assertEqual(agent.events, ["action", "reward"])
        self.assertIsNotNone(agent.action_colloids)
        self.assertIsNotNone(agent.reward_colloids)
        self.assertIsNot(agent.reward_colloids, agent.action_colloids)
        self.assertEqual(len(agent.reward_colloids), 3)
        self.assertTrue(
            all(isinstance(colloid, Colloid) for colloid in agent.reward_colloids)
        )
        self.assertEqual(
            agent.action_batches[0],
            [(10.0, 1.0, 0.0), (11.0, 2.0, 0.0), (12.0, 3.0, 0.0)],
        )
        self.assertEqual(
            agent.reward_batches[0],
            [(20.0, 4.0, 0.0), (21.0, 5.0, 0.0), (22.0, 6.0, 0.0)],
        )
        self.assertEqual(agent.reward_sendall_calls, [1])


if __name__ == "__main__":
    ut.main()

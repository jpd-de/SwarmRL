"""
Various functions for operating on colloids.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from swarmrl.components.colloid import Colloid


@dataclass
class TrajectoryInformation:
    """
    Helper dataclass for training RL models.
    """

    particle_type: int
    features: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    killed: bool = False


@jax.jit
def compute_directed_force(r: jnp.ndarray, director: jnp.ndarray) -> jnp.ndarray:
    """
    Compute a director-aligned interaction force between two colloids.

    Use this function with caution, as too big values in r can result
    in NaN-Values after the jax.grad() calculation in line 61.
    This can happen if the norm of these entries surpasses 1.5e+03.

    This uses a WCA-like potential to compute a relative interaction
    strength between two colloids. The gradient magnitude of that
    potential is then projected onto the colloid director to obtain
    a force aligned with the swimmer orientation.

    Parameters
    ----------
    r : jnp.ndarray (dimension, )
        Distance between the two colloids.
    director : jnp.ndarray (dimensions, )
        Director of the colloid.

    Returns
    -------
    force : jnp.ndarray (dimension, )
        Force magnitude aligned with the colloid director.
    """

    def _sub_compute(r):
        return (
            1e-8 + 1 / (jnp.linalg.norm(r) + 1e-8) ** 12
        )  # Add epsilon numbers to avoid the gradient to be NaN

    force_fn = jax.grad(_sub_compute)

    return jnp.linalg.norm(force_fn(r)) * director


@jax.jit
def compute_distance_matrix(set_a, set_b):
    """
    Compute a distance matrix between two sets.

    Helper function for computing the distance sets of
    colloids. This is not a commutative operation, if you
    swap a for b you will recieve a different matrix shape.

    Parameters
    ----------
    set_a : jnp.ndarray
        First set of points.
    set_b : jnp.ndarray
        Second set of points.
    """

    def _sub_compute(a, b):
        return b - a

    distance_fn = jax.vmap(_sub_compute, in_axes=(0, None))

    return distance_fn(set_a, set_b)


@jax.jit
def compute_torque(force, direction):
    """
    Compute torque from a lever arm and force vector,
    aka the torquue on a rod.

    Parameters
    ----------

    """
    return jnp.cross(direction, force)


@jax.jit
def compute_torque_partition_on_rod(
    colloid_positions, colloid_directors, rod_positions, rod_directions
):
    """
    Compute normalized torque-magnitude weights for colloids acting on a rod.

    Parameters
    ----------
    colloid_positions : jnp.ndarray (n_colloids, 3)
        Positions of the colloids.
    colloid_directors : jnp.ndarray (n_colloids, 3)
        Directors of the colloids.
    rod_positions : jnp.ndarray (rod_particles, 3)
        Positions of the rod particles.
    rod_directions : jnp.ndarray (rod_particles, 3)
        Directors of the rod particles.

    Returns
    -------
    weights : jnp.ndarray (n_colloids,)
        Scalar partition weights normalized to sum to 1.
    """
    # (n_colloids, rod_particles, 3)
    distance_matrix = compute_distance_matrix(colloid_positions, rod_positions)

    # Force on the rod
    rod_map_fn = jax.vmap(
        compute_directed_force, in_axes=(0, None)
    )  # map over rod particles
    colloid_map_fn = jax.vmap(rod_map_fn, in_axes=(0, 0))  # map over colloids

    # (n_colloids, rod_particles, 3)
    forces = colloid_map_fn(distance_matrix, colloid_directors)

    # Compute torques
    colloid_rod_map = jax.vmap(compute_torque, in_axes=(0, 0))
    colloid_only_map = jax.vmap(colloid_rod_map, in_axes=(0, None))

    torques = colloid_only_map(forces, rod_directions)
    net_rod_torque = torques.sum(axis=1)
    torque_magnitude = jnp.linalg.norm(net_rod_torque, axis=-1) + 1e-8
    normalization_factors = torque_magnitude.sum()
    torque_partition = torque_magnitude / normalization_factors

    return torque_partition


@jax.jit
def compute_rod_particle_distances(rod_positions):
    """
    Compute the vectors from the rod center to each rod particle.

    Parameters
    ----------
    rod_positions : jnp.ndarray (rod_particles, 3)
        Positions of all the rod particles.
    """

    def _sub_compute(a, b):
        return b - a

    distance_fn = jax.vmap(_sub_compute, in_axes=(0, None))
    rod_center = rod_positions[0]

    return distance_fn(rod_positions, rod_center)


@jax.jit
def compute_normalized_rod_torques(rod_positions, colloid_directors, colloid_positions):
    """
    Compute normalized signed torque contributions on a rod.

    Parameters
    ----------
    rod_positions : jnp.ndarray (rod_particles, 3)
        Positions of the rod particles.
    colloid_directors : jnp.ndarray (n_colloids, 3)
        Directors of the colloids.
    colloid_positions : jnp.ndarray (n_colloids, 3)
        Positions of the colloids.

    Returns
    -------
    torques : jnp.ndarray (n_colloids, 3)
        Signed per-colloid torque vectors normalized by the total torque
        magnitude across all colloids.
    """
    # (n_colloids, rod_particles, 3)
    distance_matrix = compute_distance_matrix(colloid_positions, rod_positions)

    # Force on the rod
    rod_map_fn = jax.vmap(
        compute_directed_force, in_axes=(0, None)
    )  # map over rod particles
    colloid_map_fn = jax.vmap(rod_map_fn, in_axes=(0, 0))  # map over colloids

    # (n_colloids, rod_particles, 3)
    forces = colloid_map_fn(distance_matrix, colloid_directors)

    # Compute torques
    colloid_rod_map = jax.vmap(compute_torque, in_axes=(0, 0))
    colloid_only_map = jax.vmap(colloid_rod_map, in_axes=(0, None))

    directions = compute_rod_particle_distances(
        rod_positions
    )  # Calculate the r vectors between the middle of the rod and each colloid.
    torques = colloid_only_map(
        forces, directions
    )  # This is used for the torque formula: T = r x F
    net_rod_torque = torques.sum(axis=1)
    torque_magnitude = jnp.linalg.norm(net_rod_torque, axis=-1) + 1e-8
    normalization_factors = torque_magnitude.sum()
    normalized_rod_torques = net_rod_torque / normalization_factors
    return normalized_rod_torques


def get_colloid_indices(colloids: List["Colloid"], p_type: int) -> List[int]:
    """
    Get the indices of the colloids in the observable of a specific type.

    Parameters
    ----------
    colloids : List[Colloid]
            List of colloids from which to get the indices.
    p_type : int
            Type of the colloids to get the indices for.


    Returns
    -------
    indices : List[int]
            List of indices for the colloids of a particular type.
    """
    indices = []
    for i, colloid in enumerate(colloids):
        if colloid.type == p_type:
            indices.append(i)

    return indices

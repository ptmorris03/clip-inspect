import jax.numpy as jnp
import jax
from functools import partial


def random_points(prng_key, count, dims):
    return jax.random.normal(prng_key, (count, dims))


def mesh1d_points(count, batch_size):
    assert count % batch_size == 0

    return NotImplementedError


def mesh2d_points(*args):
    return NotImplementedError

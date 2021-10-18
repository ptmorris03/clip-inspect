import jax.numpy as jnp
import jax
from jax import jacfwd
from functools import partial


def random_points(prng_key, count, dims):
    return jax.random.normal(prng_key, (count, dims))


def mesh1d_points(count, batch_size):
    assert count % batch_size == 0

    return NotImplementedError


def mesh2d_points(*args):
    return NotImplementedError


def pnorm(f, p=2):
    def _pnorm(x):
        y = f(x)
        return jnp.linalg.norm(y, ord=p, axis=-1)
    return _pnorm


def residual(f):
    def _residual(x):
        y = f(x)
        return y + x
    return _residual


jacobian = jacfwd

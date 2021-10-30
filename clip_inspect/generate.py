import jax.numpy as jnp
import jax
from itertools import product


def random_points(prng_key, count, dims):
    shape = (count, dims)
    if type(count) in [tuple, list]:
        shape = (*count, dims)
    return jax.random.normal(prng_key, shape)


def mesh_line1d(density=300, limit=1):
    if type(limit) is int:
        limit = [-limit, limit]

    return jnp.linspace(limit[0], limit[1], density)


def mesh_plane2d(density=300, limit=1):
    if type(limit) is int:
        limit = [[-limit, limit], [-limit, limit]]

    points = []
    for (l1, l2) in limit:
        points.append(jnp.linspace(l1, l2, density))

    return jnp.stack(jnp.meshgrid(*points), axis=-1)


def stratified_iterator(tensor, num_batches, axis=None):
    if axis is None:
        axis = list(range(len(tensor.shape) - 1))
    if type(axis) is int:
        axis = [axis]

    ranges = [range(num_batches)] * len(axis)
    for idxs in product(*ranges):
        slices = []
        axis_counter = 0
        for t_idx in range(len(tensor.shape)):
            if t_idx in axis:
                slices.append(slice(idxs[axis_counter], None, num_batches))
                axis_counter += 1
            else:
                slices.append(slice(tensor.shape[t_idx]))
        yield tuple(slices)
    


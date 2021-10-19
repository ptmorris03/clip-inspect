from clip_inspect.weights import load
from clip_inspect.model import MLP
from clip_inspect.generate import random_points, mesh_plane2d, stratified_iterator
from clip_inspect.transforms import pnorm, loop, residual
import jax.numpy as jnp
import jax
from tqdm import tqdm
from jax import vmap, jit
from functools import partial
import matplotlib.pyplot as plt


state_dict, info = load("ViT-B-32")
mlp = MLP(state_dict, "transformer.resblocks.0")
#points = random_points(mlp.prng_key, 1000, 512)
points = mesh_plane2d(density=100, limit=5)
out = jnp.zeros(points.shape[:-1])
for idxs in tqdm(stratified_iterator(points, 5)):
    idxs = idxs[:-1]

    batch = points[idxs]
    batch_shape = batch.shape[:-1]
    batch = batch.reshape(-1, points.shape[-1])

    batch = vmap(mlp.in_project)(batch)
    batch_out = vmap(jit(pnorm(loop(residual(mlp.forward), 100))))(batch)
    
    out = jax.ops.index_update(
        out,
        jax.ops.index[idxs],
        batch_out.reshape(*batch_shape)
    )



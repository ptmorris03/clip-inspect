from clip_inspect.weights import load
from clip_inspect.model import MLP
from clip_inspect.generate import random_points, mesh_plane2d, stratified_iterator
from clip_inspect.transforms import pnorm, loop, residual, loop_collect_residual
from clip_inspect.inspect import norm01
import jax.numpy as jnp
import jax
from tqdm import tqdm
from jax import vmap, jit
from functools import partial
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.io import imsave
import numpy as np

density = 2160
num_batches = 20
limit = 3
steps = 300
layer = 8

state_dict, info = load("ViT-B-32", base_path="/code/weights/")
mlp = MLP(state_dict, F"transformer.resblocks.{layer-1}")
#points = random_points(mlp.prng_key, 1000, 512)
points = mesh_plane2d(density=density, limit=limit)
out = jax.device_put(jnp.zeros((*points.shape[:-1], steps)), jax.devices("cpu")[0])
for idxs in tqdm(stratified_iterator(points, num_batches)):
    idxs = idxs[:-1]

    batch = points[idxs]
    batch_shape = batch.shape[:-1]
    batch = batch.reshape(-1, points.shape[-1])

    batch = vmap(mlp.in_project)(batch)
    batch_out = vmap(jit(loop_collect_residual(mlp.forward, steps, lambda x: jnp.linalg.norm(x, axis=-1))))(batch)
    
    out = jax.ops.index_update(
        out,
        jax.ops.index[(*idxs, slice(steps))],
        jax.device_put(batch_out.reshape(*batch_shape, steps), jax.devices("cpu")[0])
    )

for i in range(steps):
    out_img = plt.cm.gist_rainbow(norm01(out[...,i]))[...,:3]
    out_img = (out_img * 255.99).astype(np.uint8)

    out_dir = Path(F"/code/output/")
    out_dir.mkdir(parents=True, exist_ok=True)

    imsave(Path(out_dir, F"plane2d_2norm_loop100_{i:06d}.png"), out_img)

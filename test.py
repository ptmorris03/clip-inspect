from clip_inspect.weights import load
from clip_inspect.model import MLP
from clip_inspect.generate import random_points, mesh_plane2d, stratified_iterator
from clip_inspect.transforms import pnorm, residual, collect
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
num_batches = 40
limit = 3
steps = 300
layer = 8

out_dir = Path(F"/code/output/")
workspace_dir = Path(out_dir, "workspace/")

template_name = F"plane2d_2norm_{layer}_{density}_{steps}"
    
out_dir.mkdir(parents=True, exist_ok=True)
workspace_dir.mkdir(parents=True, exist_ok=True)
state_dict, info = load("ViT-B-32", base_path="/code/weights/")
mlp = MLP(state_dict, F"transformer.resblocks.{layer-1}")
#points = random_points(mlp.prng_key, 1000, 512)
points = mesh_plane2d(density=density, limit=limit)
out = np.zeros(points.shape[:-1])
for frame_idx in range(steps):
    array_path = Path(workspace_dir, F"{template_name}_{frame_idx:06d}.npy")
    np.save(array_path, out)

f_in = vmap(mlp.in_project)
f_step = vmap(jit(collect(residual(mlp.forward), lambda x: jnp.linalg.norm(x, axis=-1))))

for idxs in tqdm(stratified_iterator(points, num_batches)):
    idxs = idxs[:-1]

    batch = points[idxs]
    batch_shape = batch.shape[:-1]
    batch = batch.reshape(-1, points.shape[-1])
    batch = f_in(batch)

    for frame_idx in range(steps):
        batch, batch_out = f_step(batch)

        array_path = Path(workspace_dir, F"{template_name}_{frame_idx:06d}.npy")
        out = np.load(array_path)
        out = jax.ops.index_update(
            out,
            jax.ops.index[idxs],
            batch_out.reshape(*batch_shape)
        )
        np.save(array_path, out)

for frame_idx in range(steps):
    array_path = Path(workspace_dir, F"{template_name}_{frame_idx:06d}.npy")
    out = np.load(array_path)

    out_img = plt.cm.gist_rainbow(norm01(out))[...,:3]
    out_img = (out_img * 255.99).astype(np.uint8)

    imsave(Path(out_dir, F"{template_name}_{frame_idx:06d}.png"), out_img)

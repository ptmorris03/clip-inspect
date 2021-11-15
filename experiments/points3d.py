from clip_inspect.weights import load
from clip_inspect.model import MLP
from clip_inspect.generate import random_points
from clip_inspect.transforms import collect, vnewton, jacobian, residual, collect_residual, loop, loop_collect_residual
from clip_inspect.inspect import norm01, polar_nd
import jax.numpy as jnp
import jax
from tqdm import tqdm
from jax import vmap, pmap, jit
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
import numpy as np


repeats = 100
n_points = 10000
pre_steps = 1000
steps = 100
layer = 7


#### LOAD THE MODEL
state_dict, info = load("ViT-B-32", base_path="/code/weights/")
mlp = MLP(state_dict, F"transformer.resblocks.{layer-1}")


#### THESE FUNCTIONS DO ALL THE WORK
def collect_fn(point):
    xyz = mlp.out_project()(point)
    angle1 = jnp.arctan2(xyz[...,1], xyz[...,0])
    angle2 = jnp.arctan2(jnp.linalg.norm(xyz[...,:2], axis=-1), xyz[...,2])
    r = jnp.linalg.norm(point, axis=-1)
    return jnp.stack([
        r * jnp.cos(angle1) * jnp.sin(angle2),
        r * jnp.sin(angle1) * jnp.sin(angle2),
        r * jnp.cos(angle2)
    ], axis=-1)


def loop_fn(point):
    point = mlp.in_project(point)
    
    #pre steps
    point = loop(mlp.forward, pre_steps)(point)

    return loop_collect_residual(mlp.forward, steps, collect_fn)(point)


#### ITERATE
out_path = Path(F"/code/output/points3d/")
out_path.mkdir(exist_ok=True)

f = pmap(vmap(jit(loop_fn)))

all_coords = np.zeros((n_points * repeats, steps, 3), dtype=np.float32)

key = mlp.prng_key
for r_idx in tqdm(range(repeats)):
    key, subkey = jax.random.split(key)
    points = random_points(subkey, (n_points), 512).reshape(2, n_points // 2, 512)
    coords = f(points)
    coords = coords.reshape(-1, steps, 3)
    key, subkey = jax.random.split(key)
    all_coords[r_idx * n_points : (r_idx + 1) * n_points] = jax.random.permutation(subkey, coords)

print(all_coords.shape)
np.save(Path(out_path, F"layer{layer}_spherical.npy"), all_coords)

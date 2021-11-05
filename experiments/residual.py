from clip_inspect.weights import load
from clip_inspect.model import MLP
from clip_inspect.generate import random_points
from clip_inspect.transforms import loop, residual, loop_collect, loop_collect_residual
from clip_inspect.inspect import norm01, polar_nd, angle_nd
import jax.numpy as jnp
import jax
from tqdm import tqdm
from jax import vmap, pmap, jit
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from pathlib import Path
import numpy as np
from PIL import Image
import os


n_points = 1000
pre_steps = 1000
steps = 1000
layer = 7


#### LOAD THE MODEL
state_dict, info = load("ViT-B-32", base_path="/code/weights/")
mlp = MLP(state_dict, F"transformer.resblocks.{layer-1}")


#### FUNCTIONS

#pre_fn = loop(mlp.forward, pre_steps)
main_fn = loop_collect(residual(mlp.forward), steps, polar_nd)

@pmap
@vmap
@jit
def f(point):
    point = mlp.in_project(point)
    #point = pre_fn(point)
    return main_fn(point)


#### RUN
points = random_points(mlp.prng_key, n_points, 512)
coords = np.zeros((n_points, steps, 2))
batch = points.reshape(2, -1, 512)
coords[:] = f(batch).reshape(n_points, steps, 2)


#### PLOT
fig, ax = plt.subplots(figsize=(20, 20))
c = np.linspace(0,1, steps)
for i in tqdm(range(n_points), "plot"):
    plt.scatter(
        coords[i,:,1],
        coords[i,:,0],
        s=.1,
        c=c
    )
out_path = Path("/code/output/residual/")
out_path.mkdir(exist_ok=True)
plt.xscale('log')
fig.savefig(Path(out_path, F"polarnd_layer{layer}.png"))
plt.close()

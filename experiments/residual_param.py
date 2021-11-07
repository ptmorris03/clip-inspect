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

param_steps = 100
n_points = 1000
pre_steps = 2
steps = 1000
layer = 7


#### LOAD THE MODEL
state_dict, info = load("ViT-B-32", base_path="/code/weights/")
mlp = MLP(state_dict, F"transformer.resblocks.{layer-1}")


#### FUNCTIONS
def forward_residual_param(f, alpha):
    def _forward(x):
        return f(x) + alpha * x
    return _forward


def get_function(param_fn):
    def _f(point):
        point = mlp.in_project(point)
        point = loop(param_fn, pre_steps)(point)
        return loop_collect(param_fn, steps, polar_nd)(point)
    return pmap(vmap(jit(_f)))

alphas = jnp.linspace(0, 1, param_steps)
for frame_idx, alpha in tqdm(enumerate(alphas)):
    #### RUN
    points = random_points(mlp.prng_key, n_points, 512)
    coords = np.zeros((n_points, steps, 2))
    batch = points.reshape(2, -1, 512)

    param_fn = forward_residual_param(mlp.forward, alpha)
    f = get_function(param_fn)
    coords[:] = f(batch).reshape(n_points, steps, 2)


    #### PLOT
    fig, ax = plt.subplots(figsize=(20, 20))
    c = np.linspace(0,1, steps)
    for i in range(n_points):
        plt.scatter(
            coords[i,:,1],
            coords[i,:,0],
            s=.1,
            c=c
        )
    out_path = Path(F"/code/output/residual_param/layer{layer}/")
    out_path.mkdir(exist_ok=True, parents=True)
    plt.xscale('log')
    fig.savefig(Path(out_path, F"{frame_idx:06d}.png"))
    plt.close()
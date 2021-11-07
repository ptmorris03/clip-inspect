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

param_steps = 50
n_points = 1000
pre_steps = 1000
steps = 1000
layer = 7


#### LOAD THE MODEL
state_dict, info = load("ViT-B-32", base_path="/code/weights/")
mlp = MLP(state_dict, F"transformer.resblocks.{layer-1}")


#### FUNCTIONS
def residual_param(f, alpha):
    def _f(x):
        y = f(x)
        return y + alpha * x
    return _f


def loop_collect_residual_param(f, n, alpha, collect_f):
    def _f(x, _):
        y = f(x)
        return y + alpha * x, collect_f(y)
    _f = partial(jax.lax.scan, _f, xs=None, length=n)
    def _burn_arg(x):
        return _f(x)[1]
    return _burn_arg


def get_function(alpha):
    def _f(point):
        point = mlp.in_project(point)
        point = loop(residual_param(mlp.forward, alpha), pre_steps)(point)
        return loop_collect_residual_param(mlp.forward, steps, alpha, polar_nd)(point)
    return pmap(vmap(jit(_f)))

alphas = jnp.linspace(0, 1, param_steps)
for frame_idx, alpha in tqdm(enumerate(alphas)):
    #### RUN
    points = random_points(mlp.prng_key, n_points, 512)
    coords = np.zeros((n_points, steps, 2))
    batch = points.reshape(2, -1, 512)

    f = get_function(alpha)
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
    plt.xlim(1, 9)
    plt.ylim(-np.pi, np.pi)
    fig.savefig(Path(out_path, F"{frame_idx:06d}.png"))
    plt.close()
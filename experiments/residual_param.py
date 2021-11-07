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
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from pathlib import Path
import numpy as np
from PIL import Image
import os
from scipy.ndimage.filters import gaussian_filter

param_steps = 10000
n_points = 10000
pre_steps = 10
steps = 2000
layer = 7
blur_sigma = 3
log_base = 10


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


def f(point, alpha):
    point = mlp.in_project(point)
    point = loop(residual_param(mlp.forward, alpha), pre_steps)(point)
    return loop_collect_residual_param(mlp.forward, steps, alpha, polar_nd)(point)

@pmap
@jit
def f2(point, alpha):
    coords = vmap(f)(point, alpha).reshape(-1, 2)
    hist = jnp.histogram2d(
        coords[:,1],
        coords[:,0],
        bins=(3840, 2160),
        range=[[2, 8], [-np.pi, np.pi]]
    )[0]
    return hist

alphas = jnp.linspace(0, 1, param_steps)
for frame_idx, alpha in tqdm(enumerate(alphas)):
    #### RUN
    points = random_points(mlp.prng_key, n_points, 512)
    coords = np.zeros((n_points * steps, 2))
    batch = points.reshape(2, -1, 512)
    _alpha = np.full((2, batch.shape[1]), alpha)

    #f = get_function(alpha)
    #coords[:] = f(batch, _alpha).reshape(n_points * steps, 2)
    hist = f2(batch, _alpha).sum(axis=0)

    #hist = np.histogram2d(
    #    coords[:,1],
    #    coords[:,0],
    #    bins=(3840, 2160),
    #    range=[[2, 8], [-np.pi, np.pi]]
    #)[0]
    hist = np.flipud(hist.T)
    #hist = gaussian_filter(hist, sigma=blur_sigma)
    #hist = np.tanh(np.log(blurred + 1))
    
    #hist = np.tanh(hist)
    #hist = gaussian_filter(hist, sigma=blur_sigma)

    blurred = gaussian_filter(hist, sigma=blur_sigma)
    #zero_mask = hist == 0
    #hist[zero_mask] = blurred[zero_mask]
    hist = np.maximum(hist, blurred)
    #hist = np.tanh(hist)
    hist = np.tanh(np.log(hist + 1) / np.log(log_base))

    hist = plt.cm.inferno(norm01(hist))


    #### PLOT
    mpl.rcParams['font.size'] = 24
    mpl.rcParams['text.color'] = 'white'
    mpl.rcParams['axes.labelcolor'] = 'white'
    mpl.rcParams['xtick.color'] = 'white'
    mpl.rcParams['ytick.color'] = 'white'
    mpl.rcParams['axes.facecolor'] = 'black'
    mpl.rcParams['savefig.facecolor'] = 'black'
    fig, ax = plt.subplots(figsize=(32, 18))
    plt.imshow(hist, aspect='auto', extent=[2, 8, -np.pi, np.pi])
    #c = np.linspace(0,1, steps)
    #for i in range(n_points):
    #    plt.scatter(
    #        coords[i,:,1],
    #        coords[i,:,0],
    #        s=.1,
    #        c=c
    #    )
    out_path = Path(F"/code/output/residual_param/layer{layer}/")
    out_path.mkdir(exist_ok=True, parents=True)
    plt.xlabel("Magnitude (2-Norm)")
    plt.ylabel("Angle (Softmax Axis Circular Mean)")
    plt.title(F"Histogram of f(x(t))     |     x(t+1) = f(x(t)) + α * x(t)     |     α = {alpha:0.04f}")
    plt.tight_layout()
    fig.savefig(Path(out_path, F"{frame_idx:06d}.png"), dpi=120, pad_inches=0.0)
    plt.close()
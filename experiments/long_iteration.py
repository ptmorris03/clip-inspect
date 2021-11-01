from clip_inspect.weights import load
from clip_inspect.model import MLP
from clip_inspect.generate import random_points
from clip_inspect.transforms import collect, vnewton, jacobian, residual, collect_residual
from clip_inspect.inspect import norm01, polar_nd
import jax.numpy as jnp
import jax
from tqdm import tqdm
from jax import vmap, pmap, jit
from functools import partial
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from PIL import Image

n_accum = 10
n_points = 4000
pre_steps = 1000
steps = 10000
layer = 7

hist_bins = [1080, 1920]

#for layer 7
#hist_range = [[3.15, 4.78], [0.632, 0.705]]
hist_range = [[-np.pi, np.pi], [3, 7]]

state_dict, info = load("ViT-B-32", base_path="/code/weights/")
mlp = MLP(state_dict, F"transformer.resblocks.{layer-1}")

def f_collect(x):
    return polar_nd(x)

f_in = pmap(vmap(mlp.in_project))
f_forward = pmap(vmap(jit(collect(mlp.forward, f_collect))))
f_jac = pmap(vmap(jit(jacobian(mlp.forward))))

def lyapunov_dimension(J):
    eigvals = jnp.flip(jnp.sort(jnp.linalg.eigvals(J)))
    cumsum = jnp.cumsum(eigvals)
    j = (cumsum >= 0).sum()
    return j + cumsum[j - 1] / jnp.abs(eigvals[j])
f_lldimension = vmap(jit(lyapunov_dimension, backend='cpu'))

def hist2d(coords):
    return jnp.histogram2d(
        coords[...,0].reshape(-1),
        coords[...,1].reshape(-1),
        bins=hist_bins,
        range=hist_range,
        density=True
    )[0]
hist2d = vmap(jit(hist2d))

def batch_hist2d(coords):
    return hist2d(coords).sum(axis=0)
batch_hist2d = pmap(jit(batch_hist2d))

key = mlp.prng_key
hist = None
for a_idx in tqdm(range(n_accum)):
    key, subkey = jax.random.split(key)
    out = np.zeros((2 * n_points, steps, 2))
    points = random_points(subkey, 2 * n_points, 512)
    points = points.reshape(2, n_points, 512)
    points = f_in(points)
    for s_idx in tqdm(range(pre_steps), leave=False):
        points, coords = f_forward(points)
    for s_idx in tqdm(range(steps), leave=False):
        points, coords = f_forward(points)
        #J = f_jac(points).reshape(2 * n_points, 512, 512)
        #d = f_lldimension(J)

        out[:,s_idx,:] = coords.reshape(-1, 2)
    out = out.reshape(2, -1, 1000, steps, 2)

    #print(out.reshape(-1,2).min(axis=0), out.reshape(-1,2).max(axis=0))

    _hist = batch_hist2d(out).sum(axis=0)
    if hist is None:
        hist = _hist
    else:
        hist += _hist

hist_norm = norm01(jnp.maximum(0, jnp.log(hist + 1e-9)))
im = (plt.cm.inferno(hist_norm)[...,:3] * 255.99).astype(np.uint8)
im = Image.fromarray(im)
im.save(F"/code/output/map_attractor_layer{layer}.png")

#fig, ax = plt.subplots(figsize=(20, 20))
#c = np.linspace(0,1, steps)
#for i in range(2 * n_points):
#    plt.scatter(
#        out[i,:,0],
#        out[i,:,1],
#        s=.1,
#        #c=c
#    )
#fig.savefig("/code/output/long_iteration2.png")

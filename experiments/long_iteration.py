from clip_inspect.weights import load
from clip_inspect.model import MLP
from clip_inspect.generate import random_points
from clip_inspect.transforms import collect, vnewton, jacobian, residual, collect_residual
from clip_inspect.inspect import norm01, axis_color
import jax.numpy as jnp
import jax
from tqdm import tqdm
from jax import vmap, pmap, jit
from functools import partial
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

n_points = 2000
pre_steps = 1000
steps = 1
layer = 7

state_dict, info = load("ViT-B-32", base_path="/code/weights/")
mlp = MLP(state_dict, F"transformer.resblocks.{layer-1}")

def f_collect(x):
    return jnp.linalg.norm(x, axis=-1), axis_color(x)

f_in = pmap(vmap(mlp.in_project))
f_forward = pmap(vmap(jit(collect(mlp.forward, f_collect))))
f_jac = pmap(vmap(jit(jacobian(mlp.forward))))

def lyapunov_dimension(J):
    eigvals = jnp.flip(jnp.sort(jnp.linalg.eigvals(J).real))
    cumsum = jnp.cumsum(eigvals)
    j = (cumsum >= 0).sum()
    return j + cumsum[j - 1] / jnp.abs(eigvals[j])
f_lldimension = vmap(jit(lyapunov_dimension, backend='cpu'))

out = np.zeros((2 * n_points, steps, 2))
points = random_points(mlp.prng_key, 2 * n_points, 512)
points = points.reshape(2, n_points, 512)
points = f_in(points)
for s_idx in tqdm(range(pre_steps)):
    points, (out_norm, out_axis) = f_forward(points)
for s_idx in tqdm(range(steps)):
    points, (out_norm, out_axis) = f_forward(points)
    #J = f_jac(points).reshape(2 * n_points, 512, 512)
    #d = f_lldimension(J)

    out[:,s_idx,0] = out_norm.reshape(-1)
    out[:,s_idx,1] = out_axis.reshape(-1)
    
fig, ax = plt.subplots(figsize=(5, 5))
c = np.linspace(0,1, steps)
for i in range(2 * n_points):
    plt.scatter(
        out[i,:,0],
        out[i,:,1],
        s=1,
        c=c
    )
fig.savefig("/code/output/long_iteration2.png")

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

n_points = 2
steps = 30
layer = 8

state_dict, info = load("ViT-B-32", base_path="/code/weights/")
mlp = MLP(state_dict, F"transformer.resblocks.{layer-1}")


f_in = pmap(vmap(mlp.in_project))
f_forward = pmap(vmap(jit(collect_residual(mlp.forward, mlp.out_project(n_components=2)))))
f_jac = pmap(vmap(jit(jacobian(mlp.forward))))

def lyapunov_dimension(J):
    eigvals = jnp.flip(jnp.sort(jnp.abs(jnp.linalg.eigvals(J))))
    cumsum = jnp.cumsum(eigvals)
    j = (cumsum >= 0).sum()
    return j + cumsum[j - 1] / jnp.abs(eigvals[j])
f_lldimension = vmap(jit(lyapunov_dimension, backend='cpu'))

out = np.zeros((2 * n_points, steps, 3))
points = random_points(mlp.prng_key, 2 * n_points, 512)
points = points.reshape(2, n_points, 512)
points = f_in(points)
for s_idx in tqdm(range(steps)):
    points, out_coords = f_forward(points)
    J = f_jac(points).reshape(2 * n_points, 512, 512)
    d = f_lldimension(J)

    out[:,s_idx,0] = d.reshape(-1)
    out[:,s_idx,1:3] = out_coords.reshape(-1, 2)
    
fig, ax = plt.subplots()

for i in range(n_points):
    plt.scatter(
        out[i,:,1], 
        out[i,:,0], 
        s=1,
        c=out[i,:,2]
    )
fig.savefig("/code/output/dimension.png")

from clip_inspect.weights import load
from clip_inspect.model import MLP
from clip_inspect.generate import random_points
from clip_inspect.transforms import vnewton, jacobian, residual, collect_residual
from clip_inspect.inspect import norm01, axis_color
import jax.numpy as jnp
import jax
from tqdm import tqdm
from jax import vmap, pmap, jit
from functools import partial
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

n_points = 10
steps = 30
layer = 8

state_dict, info = load("ViT-B-32", base_path="/code/weights/")
mlp = MLP(state_dict, F"transformer.resblocks.{layer-1}")

f_in = pmap(vmap(mlp.in_project))
f_forward = pmap(vmap(jit(collect_residual(mlp.forward, axis_color))))
f_jac = pmap(vmap(jit(jacobian(residual(mlp.forward)))))

def lyapunov_dimension(J):
    eigvals = jnp.flip(jnp.sort(jnp.linalg.eigvals(J).real))
    cumsum = jnp.cumsum(eigvals)
    j = jnp.argmax(cumsum >= 0)
    return j + cumsum[j] / jnp.abs(eigvals[j + 1])
f_lldimension = vmap(jit(lyapunov_dimension, backend='cpu'))

out = np.zeros((2 * n_points, steps, 2))
points = random_points(mlp.prng_key, 2 * n_points, 512)
points = points.reshape(2, n_points, 512)
points = f_in(points)
for s_idx in tqdm(range(steps)):
    points, color = f_forward(points)
    J = f_jac(points).reshape(2 * n_points, 512, 512)
    d = f_lldimension(J)

    out[:,s_idx,0] = d.reshape(-1)
    out[:,s_idx,1] = color.reshape(-1)
    
fig, ax = plt.subplots()

for i in range(n_points):
    plt.scatter(
        out[i,:,0], 
        out[i,:,1], 
        s=1,
    )
fig.savefig("/code/output/dimension.png")

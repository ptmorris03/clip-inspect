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

n_points = 100
steps = 300
layer = 8

state_dict, info = load("ViT-B-32", base_path="/code/weights/")
mlp = MLP(state_dict, F"transformer.resblocks.{layer-1}")

f_in = pmap(vmap(mlp.in_project))
f_forward = pmap(vmap(jit(collect_residual(mlp.forward, axis_color))))


def slogdet_trace(f):
    def _slogdet_trace(x):
        y = f(x)
        sign, logdet = jnp.linalg.slogdet(y)
        trace = jnp.trace(y)
        return sign, logdet, trace
    return _slogdet_trace
f_eval = pmap(vmap(jit(slogdet_trace(jacobian(mlp.forward)))))

def simplify_slogdet(sign, logdet):
    np_sign = np.array(sign).astype(np.float64)
    np_det = np.exp(np.array(logdet).astype(np.float64))
    return np_sign * np_det

out = np.zeros((2 * n_points, steps, 3))
points = random_points(mlp.prng_key, 2 * n_points, 512)
points = points.reshape(2, n_points, 512)
points = f_in(points)
for s_idx in tqdm(range(steps)):
    sign, logdet, trace = f_eval(points)
    points, color = f_forward(points)

    out[:,s_idx,0] = logdet.reshape(-1)
    out[:,s_idx,1] = trace.reshape(-1)
    out[:,s_idx,2] = color.reshape(-1)
    
fig, ax = plt.subplots()

for i in range(n_points):
    plt.scatter(
        out[i,:,0], 
        out[i,:,1], 
        s=.1, 
        c=out[i,:,2], 
        cmap='gist_rainbow'
    )
fig.savefig("/code/output/trajectories.png")

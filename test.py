from clip_inspect.weights import load
from clip_inspect.model import MLP
from clip_inspect.generate import random_points
from clip_inspect.transforms import pnorm, loop, residual
import jax.numpy as jnp
from jax import vmap, jit
from functools import partial


state_dict, info = load("ViT-B-32")
mlp = MLP(state_dict, "transformer.resblocks.0")
points = random_points(mlp.prng_key, 1000, 512)
points = vmap(mlp.in_project)(points)
all_points = vmap(jit(pnorm(loop(residual(mlp.forward), 100))))(points)
print(all_points.shape)

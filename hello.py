from clip_inspect.weights import load
from clip_inspect.model import MLP
from clip_inspect.inspect import random_points
import jax.numpy as jnp
from jax import vmap, jit
from functools import partial


state_dict, info = load("ViT-B-32")
mlp = MLP(state_dict, "transformer.resblocks.0")
points = random_points(mlp.prng_key, 1000, 512)
print(info)
print(jnp.linalg.norm(vmap(jit(mlp.forward))(points), axis=-1))

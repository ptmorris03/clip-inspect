from clip_inspect.weights import load
from clip_inspect.model import MLP
from clip_inspect.generate import mesh_sphere2d
from clip_inspect.transforms import loop, collect, jacobian
from clip_inspect.inspect import norm01, polar_nd
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


mesh_size = 50 #17280
steps = 1000
l_steps = 100
layer = 7


#### LOAD THE MODEL
state_dict, info = load("ViT-B-32", base_path="/code/weights/")
mlp = MLP(state_dict, F"transformer.resblocks.{layer-1}")


#### FUNCTIONS
def lyapunov_step(point, circle):
    point = mlp.forward(point)
    J = jacobian(mlp.forward)(point)

    ellipse = jnp.matmul(J, circle)
    circle, r = jnp.linalg.qr(ellipse, mode='complete')
    exponents = jnp.log(jnp.abs(jnp.diag(r)))

    return point, circle, exponents


def lyapunov_dimension(exponents):
    sorted = jnp.flip(jnp.sort(exponents))
    sums = jnp.cumsum(sorted)
    j = (sums >= 0).sum()
    return jax.lax.cond(
        j == 0,
        lambda _: 0.0,
        lambda _: _,
        operand=j + sums[j - 1] / jnp.abs(sorted[j])
    )


def lyapunov_exponents(point):
    def step_fn(inputs, _):
        point, circle = inputs
        point, circle, exponents = lyapunov_step(point, circle)
        return (point, circle), exponents
    circle = jnp.eye(point.shape[-1])
    exponents = jax.lax.scan(
        step_fn, 
        (point, circle), 
        xs=None, 
        length=l_steps
    )[1]
    
    #cumulative mean exponents and dimension
    exponents = exponents.sum(axis=0) / l_steps
    dimension = lyapunov_dimension(exponents)

    return dimension

@pmap
@vmap
@jit
def f(point):
    point = mlp.in_project(point)
    return collect(loop(mlp.forward, steps), lyapunov_exponents)(point)[1]


#### RUN
mesh = mesh_sphere2d(mesh_size)
coords = np.zeros((mesh_size, mesh_size))
for i in tqdm(range(mesh.shape[0])):
    batch = mesh[i].reshape(2, -1, 3)
    coords[i] = f(batch).reshape(-1)


#### SAVE
#hsv = jnp.stack([
#    norm01(coords[:,:,0]),
#    jnp.ones((mesh_size, mesh_size)),
#    norm01(coords[:,:,1]) * (1 - min_brightness) + min_brightness
#], axis=-1)
#rgb = (hsv_to_rgb(hsv) * 255.99).astype(np.uint8)
rgb = plt.cm.viridis(norm01(coords))
im = Image.fromarray((rgb * 255.99).astype(np.uint8))

out_path = Path("/code/output/basins_of_attraction/")
out_path.mkdir(exist_ok=True)
out_file = Path(out_path, F"basins_layer{layer}_dimension.png")
out_file = open(out_file, 'wb')
im.save(out_file, 'PNG')
os.fsync(out_file)
out_file.close()

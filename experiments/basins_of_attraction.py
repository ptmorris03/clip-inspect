from clip_inspect.weights import load
from clip_inspect.model import MLP
from clip_inspect.generate import mesh_sphere2d
from clip_inspect.transforms import loop, collect
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


mesh_size = 16000 #17280
steps = 1
min_brightness = 0.5
layer = 7


#### LOAD THE MODEL
state_dict, info = load("ViT-B-32", base_path="/code/weights/")
mlp = MLP(state_dict, F"transformer.resblocks.{layer-1}")


#### FUNCTIONS
@pmap
@vmap
@jit
def f(point):
    point = mlp.in_project(point)
    return collect(loop(mlp.forward, steps), polar_nd)(point)[1]


#### RUN
mesh = mesh_sphere2d(mesh_size)
coords = np.zeros((mesh_size, mesh_size, 2))
for i in tqdm(range(mesh.shape[0])):
    batch = mesh[i].reshape(2, -1, 3)
    coords[i] = f(batch).reshape(-1, 2)


#### SAVE
hsv = jnp.stack([
    norm01(coords[:,:,0]),
    jnp.ones((mesh_size, mesh_size)),
    norm01(coords[:,:,1]) * (1 - min_brightness) + min_brightness
], axis=-1)
rgb = (hsv_to_rgb(hsv) * 255.99).astype(np.uint8)
im = Image.fromarray(rgb)

out_path = Path("/code/output/basins_of_attraction/")
out_path.mkdir(exist_ok=True)
out_file = Path(out_path, F"basins_layer{layer}_test.png")
out_file = open(out_file, 'wb')
im.save(out_file, 'PNG')
os.fsync(out_file)
out_file.close()

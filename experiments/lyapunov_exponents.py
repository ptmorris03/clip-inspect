from clip_inspect.weights import load
from clip_inspect.model import MLP
from clip_inspect.generate import random_points
from clip_inspect.transforms import collect, vnewton, jacobian, residual, collect_residual, loop
from clip_inspect.inspect import norm01, polar_nd
import jax.numpy as jnp
import jax
from tqdm import tqdm
from jax import vmap, pmap, jit
from functools import partial
import matplotlib.pyplot as plt
from mlp_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
import numpy as np


n_points = 10
pre_steps = 1000
steps = 10000
layer = 7


#### LOAD THE MODEL
state_dict, info = load("ViT-B-32", base_path="/code/weights/")
mlp = MLP(state_dict, F"transformer.resblocks.{layer-1}")


#### THESE FUNCTIONS DO ALL THE WORK
def lyapunov_step(point, circle):
    point, circle = inputs
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
    return j + sums[j - 1] / jnp.abs(sorted[j])


def lyapunov_exponents(point):
    point = mlp.in_project(point)
    
    #pre steps
    point = loop(mlp.forward, pre_steps)(point)

    #main loop
    def step_fn(inputs, _):
        point, circle = inputs
        point, circle, exponents = lyapunov_step(point, circle)
        coords = polar_nd(point)
        return (point, circle), (exponents, coords)
    circle = jnp.eye(point.shape[-1])
    exponents, coords = jax.lax.scan(
        step_fn, 
        (point, circle), 
        xs=None, 
        length=steps
    )[1]
    
    #cumulative mean exponents and dimension
    denoms = jnp.arange(1, steps + 1).reshape(-1, 1)
    exponents = jnp.cumsum(exponents, axis=0) / denoms
    dimension = vmap(lyapunov_dimension)(exponents)

    return exponents, dimension, coords


#### THE PLOT NEEDS TO LOOK PRETTY
def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter


class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))


#### ITERATE
f_exp = pmap(vmap(jit(lyapunov_exponents)))

points = random_points(mlp.prng_key, (2, n_points), 512)
exponents, dimension, coords = f_exp(points)

exponents = exponents.reshape(2 * n_points, steps, 512)
dimension = dimension.reshape(2 * n_points, steps)
coords = coords.reshape(2 * n_points, steps, 2)


#### PLOT AND SAVE
plt.rc('font', size=20)
out_path = Path(F"/code/output/exponent_plots{layer}/")
out_path.mkdir(exist_ok=True)
for i in tqdm(range(2 * n_points)):
    fig = plt.figure(figsize=(18, 18), dpi=120)
    
    ax = fig.add_subplot(3, 3, (1, 2))
    ax.set_title("Lyapunov Exponents")
    for e in range(512):
        color = plt.cm.gist_rainbow(e / 512)
        ax.plot(exponents[i,:,e], color=color)

    ax = fig.add_subplot(3, 3, 3)
    ax.set_title(F"Dimension ({dimension[i,-1]:.02f})")
    ax.plot(dimension[i])

    ax = fig.add_subplot(3, 3, (4, 9))
    ax.set_title("Attractor (CLIP Text Layer {layer})")
    ax.set_xlabel("Magnitude ~ 2-Norm")
    ax.set_ylabel("Angle ~ Circular Mean of Softmax(Abs(point))")
    ax.set_ylim(-np.pi, np.pi)
    ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

    scatter = ax.scatter(
        coords[i,:,1], 
        coords[i,:,0], 
        c=np.linspace(0, steps, steps)
    )
    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(scatter, cax=cax, orientation='vertical')

    fig.tight_layout()
    fig.savefig(
        Path(out_path, F"point{i:06d}.png"), 
        dpi='figure', 
        transparent=False, 
        pad_inches=0
    )
    plt.close(fig)
    
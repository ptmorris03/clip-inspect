import jax.numpy as jnp
import jax


def norm01(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def axis_color(vector):
    probs = jax.nn.softmax(jnp.abs(vector))
    weights = jnp.linspace(0, 1, vector.shape[-1], endpoint=False)
    return (probs * weights).sum(axis=-1)

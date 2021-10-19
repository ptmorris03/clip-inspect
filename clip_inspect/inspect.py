import jax.numpy as jnp


def norm01(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())
    
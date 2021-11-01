import jax.numpy as jnp
import jax


def norm01(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def magnitude(vector):
    return jnp.linalg.norm(vector, axis=-1)


def angle_nd(vector):
    angles = jnp.linspace(-jnp.pi, jnp.pi, vector.shape[-1] * 2, endpoint=False)
    angles = angles + ((angles[1] - angles[0]) / 2)
    probs = jax.nn.softmax(jnp.concatenate([-vector, vector], axis=-1))
    xcoord = (probs * jnp.cos(angles)).sum(axis=-1)
    ycoord = (probs * jnp.sin(angles)).sum(axis=-1)
    return jnp.arctan2(xcoord, ycoord)


def polar_nd(vector):
    return jnp.stack([angle_nd(vector), magnitude(vector)], axis=-1)

##HUE HISTOGRAM CLASS
#   - remembers extent (set automatically in first call)
#   - one hist for count
#   - corresponding hist for hue weight
#   - to display, hue = weight / count, intensity = count
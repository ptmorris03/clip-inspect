import jax.numpy as jnp
import jax
from jax import grad, jacfwd
from functools import partial


def pnorm(f, p=2):
    def _pnorm(x):
        y = f(x)
        return jnp.linalg.norm(y, ord=p, axis=-1)
    return _pnorm


def residual(f):
    def _residual(x):
        y = f(x)
        return y + x
    return _residual


def collect(f, collect_f):
    def _collect(x):
        y = f(x)
        return y, collect_f(y)
    return _collect


def collect_residual(f, collect_f):
    def _collect_residual(x, _):
        y = f(x)
        return y + x, collect_f(y)
    return _collect_residual


jacobian = jacfwd


def newton(f, step_size=1):
    def _newton(x):
        return x - step_size * grad(f)(x)
    return _newton


def vnewton(f, step_size=1):
    def _vnewton(x):
        y = f(x)
        J = jacobian(f)(x)
        delta_x = jnp.linalg.solve(J, y)
        return x - step_size * delta_x
    return _vnewton


def loop(f, n):
    def _f(_, x):
        return f(x)
    return partial(jax.lax.fori_loop, 0, n, _f)


def loop_collect(f, n, collect_f):
    def _f(x, _):
        y = f(x)
        return y, collect_f(y)
    _f = partial(jax.lax.scan, _f, xs=None, length=n)
    def _burn_arg(x):
        return _f(x)[1]
    return _burn_arg


def loop_collect_residual(f, n, collect_f):
    def _f(x, _):
        y = f(x)
        return y + x, collect_f(y)
    _f = partial(jax.lax.scan, _f, xs=None, length=n)
    def _burn_arg(x):
        return _f(x)[1]
    return _burn_arg
import jax
import jax.numpy as jnp
from functools import partial


@jax.jit
def jit_map_coordinates(vals, coords):
    return jax.scipy.ndimage.map_coordinates(vals, coords, order=1, mode='nearest')

@partial(jax.jit, static_argnames=['dim'])
def vals_to_coords(grids, x_vals, dim):
    """Transform values of the states to corresponding coordinates (array indices) on the grids.
    """
    intervals = jnp.asarray([grid[1] - grid[0] for grid in grids])
    low_bounds = jnp.asarray([grid[0] for grid in grids]) 
    intervals = intervals.reshape(dim, 1)
    low_bounds = low_bounds.reshape(dim, 1)
    return (x_vals - low_bounds) / intervals

@jax.jit
def lin_interp(grid, values, points):
    dim = points.shape[0]
    coords = vals_to_coords(grid, points, dim)
    # Interpolate using coordinates
    interp_vals = jit_map_coordinates(values, coords) 
    return interp_vals
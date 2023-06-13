### Code from the interpolation library 
import numpy as np
import time
from interpolation.splines import UCGrid, CGrid, nodes, eval_linear
from interp import vals_to_coords, lin_interp
import jax.numpy as jnp

if __name__ == '__main__':

    f = lambda x,y: np.sin(np.sqrt(x**2+y**2+0.00001))/np.sqrt(x**2+y**2+0.00001)

    # uniform cartesian grid
    N_points = 10
    N_grid = 10

    print('Number of interpolatation points: ', N_points)
    print('Number of grid points: ', N_points)

    grid = UCGrid((-1.0, 1.0, N_grid), (-1.0, 1.0, N_grid))
    # get grid points
    gp = nodes(grid)   # 100x2 matrix
    
    # compute values on grid points
    values = f(gp[:,0], gp[:,1]).reshape((N_grid, N_grid))

    # interpolate at many points:
    np.random.seed(10)
    points = np.random.random((N_points,2))

    time_numba = time.time()
    linear_interp_numba = eval_linear(grid, values, points) # 10000 vector
    print('Time Numba version:\n', time.time() - time_numba)

    grids = (jnp.linspace(*grid[0]), jnp.linspace(*grid[1]))
    values_jnp = jnp.asarray(values)
    points_jnp = jnp.asarray(points.T)

    time_jax = time.time()
    linear_interp_jax = lin_interp(grids, values_jnp, points_jnp).block_until_ready()
    print('Time JAX version:\n', time.time() - time_jax)

    assert jnp.allclose(linear_interp_numba, linear_interp_jax)




### Code from the interpolation library 
import numpy as np
import time
from interpolation.splines import UCGrid, CGrid, nodes, eval_linear
from interp import vals_to_coords, lin_interp
import jax.numpy as jnp

def test_linear_interp(N_points, N_grid):
    f = lambda x,y: np.sin(np.sqrt(x**2+y**2+0.00001))/np.sqrt(x**2+y**2+0.00001)

    print('Number of interpolatation points: ', N_points)
    print('Number of grid points: ', N_grid)

    grid = UCGrid((-1.0, 1.0, N_grid), (-1.0, 1.0, N_grid))
    # get grid points
    gp = nodes(grid)   # 100x2 matrix
    
    # compute values on grid points
    values = f(gp[:,0], gp[:,1]).reshape((N_grid, N_grid))

    points = np.random.random((N_points,2))

    time_start = time.time()
    linear_interp_numba = eval_linear(grid, values, points) # 10000 vector
    time_numba = time.time() - time_start
    print('Time Numba version:\n', time_numba)

    grids = (jnp.linspace(*grid[0]), jnp.linspace(*grid[1]))
    values_jnp = jnp.asarray(values)
    points_jnp = jnp.asarray(points.T)

    time_start = time.time()
    linear_interp_jax = lin_interp(grids, values_jnp, points_jnp).block_until_ready()
    time_jax = time.time() - time_start
    print('Time Numba version:\n', time_jax)
    
    print('Difference between Numba and JAX versions: ',
          jnp.linalg.norm(linear_interp_numba - linear_interp_jax))
    assert jnp.allclose(linear_interp_numba, linear_interp_jax, atol=1e-4), 'Interpolation results are not the same'
    print('-'*64)

    return time_numba, time_jax

if __name__ == '__main__':
    time_numba_list = []
    time_jax_list = []

    for i in np.arange(2, 1001, 100):
        for j in np.arange(2, 1001, 100):
            time_numba, time_jax = test_linear_interp(i, j)
            time_numba_list.append(time_numba)
            time_jax_list.append(time_jax)

    # plot the results 
    import matplotlib.pyplot as plt
    plt.plot(time_numba_list, label='Numba')
    plt.plot(time_jax_list, label='JAX')
    plt.xlabel('Trial (n)')
    plt.ylabel('Time (s)')
    plt.title('Comparison of Numba and JAX versions of linear interpolation')
    plt.savefig('linear_interp.png')
    

   




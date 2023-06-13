import numpy as np
import time
from interpolation.splines import UCGrid, nodes, eval_linear
from interp import lin_interp
import jax.numpy as jnp
import matplotlib.pyplot as plt

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
    print('Time JAX version:\n', time_jax)
    
    print('Difference between Numba and JAX versions: ',
          jnp.linalg.norm(linear_interp_numba - linear_interp_jax))
    assert jnp.allclose(linear_interp_numba, linear_interp_jax), 'Interpolation results are not the same'
    print('-'*64)

    return time_numba, time_jax

if __name__ == '__main__':
    run_time = np.array([[test_linear_interp(i, j), test_linear_interp(i, j)] 
              for i in np.arange(2, 1000, 100) for j in np.arange(2, 1000, 100)])

    # plot the results 
    plt.plot(run_time[:,0][:,0], label='Numba')
    plt.plot(run_time[:,0][:,1], label='JAX')
    plt.legend()
    plt.xlabel('# Trial')
    plt.ylabel('Time (s)')
    plt.title('Comparison of Numba and JAX versions of linear interpolation')
    plt.savefig('results/linear_interp.png')
    plt.clf()

    # plot the results 
    plt.plot(run_time[:,1][:,0], label='Numba')
    plt.plot(run_time[:,1][:,1], label='JAX')
    plt.legend()
    plt.xlabel('# Trial')
    plt.ylabel('Time (s)')
    plt.title('Comparison of Numba and JAX versions of linear interpolation (Run 2)')
    plt.savefig('results/linear_interp_run2.png')
    

   




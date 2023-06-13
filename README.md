# JAX compatible multilinear interpolation

We need fast linear interpolation in arbitrary dimensions, compatible with
`jax.jit`.

Suppose we have a function $f$ mapping $\RR^n$ to $\RR$ and we evalute $f$ on a
finite set of grid points $v_1, \ldots, v_k$, where each $v_i$ is a point in
$\RR^n$.  Let $f_i = f(v_i)$.

We now want to be able to evaluate $\hat f(x)$, which is a linear interpolation
of the grid points $v_1, \ldots, v_k$ and corresponding function values $f_i,
\ldots, f_k$, at the point $x$.

We want to do this where $n$ can be any integer.

We also want to be able to do this in a vectorized, parallelized manner, so that
we can pass a large number of evaluation points $x_1, \ldots, x_m$, and compute
the points $\hat f(x_i), \ldots, f(x_m)$ efficiently.

This is already possible in Numba, using the Econforge
[interpolation library](https://www.econforge.org/interpolation.py/):


```python
import numpy as np
from interpolation.splines import UCGrid, CGrid, nodes
from interpolation.splines import eval_linear

# function to interpolate
f = lambda x,y: np.sin(np.sqrt(x**2+y**2+0.00001))/np.sqrt(x**2+y**2+0.00001)

# uniform cartesian grid with a small number of points
grid = UCGrid((-1.0, 1.0, 10), (-1.0, 1.0, 10))

# get grid points
gp = nodes(grid)   # 100x2 matrix

# compute values on grid points
values = f(gp[:,0], gp[:,1]).reshape((10,10))

# interpolate at one point
point = np.array([0.1,0.45]) # 1d array
val = eval_linear(grid, values, point)  # float

# interpolate at many points:
points = np.random.random((10000,2))
eval_linear(grid, values, points) # 10000 vector
```

The question is, what is the most efficient way to replicate this in JAX?

Using `jax.scipy.ndimage.map_coordinates`, @junnanZ has implemented evaluation at a single point for $n=4$ in [this file](https://github.com/jstac/sdfs_via_autodiff/blob/main/code/ssy/continuous_junnan/utils.py).

We could then extend this from one `point` to `points` via `vmap`.

This discussion suggests that the vmap approach will be efficient: https://github.com/google/jax/issues/6312

Other alternatives include https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/math/batch_interp_regular_nd_grid

Are these our best options?  If so, which is best?

Once we have decided, let's 

1. test to make sure that the output agrees with the output from the Python code
   above, and
1. provide a nice interface, like the one in the Python code above.

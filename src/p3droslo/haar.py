import numpy as np
import numba


class Haar():

    def __init__(self, points, q):
        """
        Constructor for a Haar object.

        Parameters
        ----------
        points    : n by 3 array 
        q         : Maximum level of refinement (side of the cube is 2**q, number of volume elements 8**q)
        """
        # Store the maximum (finenst) level
        self.q = q
        
        # Compute size of the box
        self.xyz_min = np.min(points, axis=0)
        self.xyz_max = np.max(points, axis=0)

        # Set a tolerance (which should be smaller than the resolution of the cube)
        tol = 1.0e-3 / 2**self.q

        # Compute xyz size of the box (1 tol'th larger)
        self.xyz_L = (1.0 + tol) * (self.xyz_max - self.xyz_min)

        # Normalize the point coordinates to be in [0,1]
        self.points_normed = (points - self.xyz_min) / self.xyz_L

        # Count the number of points in each octree grid cell
        self.num = Haar.get_number_matrices(self.points_normed, self.q)

        # Compute cube (ix, iy, iz) indices in the linear representation 
        self.ids = [np.array(Haar.__cub__(np.arange(8**k, dtype=np.int64)), dtype=np.int64).T for k in range(q)]


    def map_data(self, data, interpolate=True):
        """
        Map data to the cube.
        """
        # Integrate (i.e. sum) the point data over each cell
        dat = Haar.integrate_data(self.points_normed, self.q, data)
        # Divide by the numberof points in each cell to get the average
        for k in range(self.q):
            dat[k] = np.divide(dat[k], self.num[k], out=dat[k], where=(self.num[k]!=0))
        # Interpolate empty cells in each cube
        if interpolate:
            for k in range(1, self.q):
                dat[k] = Haar.interpolate(self.ids[k], self.num[k], dat[k], dat[k-1], k)
        # Return a hierarchical representation of the data
        return dat


    @staticmethod
    @numba.njit(parallel=True)
    def get_number_matrices(points_normed, q):
        """
        Compute the number of points that live in each cell at each level.
        """
        # Create number matrices
        num = [np.zeros((2**k, 2**k, 2**k), dtype=np.int64) for k in range(q)]
        # For all levels
        for k in range(q):
            # Compute the indices of the points in the octree grid
            indices = (points_normed * 2**k).astype(np.int64)
            # Count the number of points at every index
            for ix, iy, iz in indices:
                num[k][ix, iy, iz] += 1
        # Return the number matrices
        return num

    
    @staticmethod
    @numba.njit(parallel=True)
    def integrate_data(points_normed, q, data):
        """
        Sum the data in each cell at each level.
        """
        # Create number matrices
        dat = [np.zeros((2**k, 2**k, 2**k), dtype=data.dtype) for k in range(q)]
        # For all levels
        for k in range(q):
            # Compute the indices of the points in the octree grid
            indices = (points_normed * 2**k).astype(np.int64)
            # Count the number of points at every index
            for i, (ix, iy, iz) in enumerate(indices):
                dat[k][ix, iy, iz] += data[i]
        # Return the integrated data
        return dat


    @staticmethod
    @numba.njit(parallel=True)
    def interpolate(ids, num, dat, dat_up, k):
        # For all cells in the cube at this level
        for i in numba.prange(8**k):
            # Get the cube triple index
            ix, iy, iz = ids[i]
            # If the cell at this level is empty take the value from the above level
            if (num[ix, iy, iz] == 0):
                dat[ix, iy, iz] = dat_up[ix//2, iy//2, iz//2]
        # Return data cube
        return dat
    
    
    @staticmethod
    @numba.njit(parallel=True)
    def __lin__(i, j, k):
        """
        If the binary representation of i, j, and k are given by:
            bin(i) = (..., i8, i4, i2, i1)
            bin(j) = (..., j8, j4, j2, j1)
            bin(k) = (..., k8, k4, k2, k1)
        this function returns the number, r, for which the binary repressentation is given by:
            bin(r) = (..., k8, j8, i8, k4, j4, i4, k2, j2, i2, k1, j1, i1)
        inducing an hierarchical ordering.
        """

        r = 0

        j = j << 1
        k = k << 2

        r = r + (i & 2**0)
        r = r + (j & 2**1)
        r = r + (k & 2**2)

        # Yes, this can be rewritten as a for-loop
        # Done this way since I am not sure if the compiler unrolls loops.
        # Should be tested!

        i = i << 2
        j = j << 2
        k = k << 2

        r = r + (i & 2**3)
        r = r + (j & 2**4)
        r = r + (k & 2**5)

        i = i << 2
        j = j << 2
        k = k << 2

        r = r + (i & 2**6)
        r = r + (j & 2**7)
        r = r + (k & 2**8)

        i = i << 2
        j = j << 2
        k = k << 2

        r = r + (i & 2**9)
        r = r + (j & 2**10)
        r = r + (k & 2**11)

        i = i << 2
        j = j << 2
        k = k << 2

        r = r + (i & 2**12)
        r = r + (j & 2**13)
        r = r + (k & 2**14)

        i = i << 2
        j = j << 2
        k = k << 2

        r = r + (i & 2**15)
        r = r + (j & 2**16)
        r = r + (k & 2**17)

        i = i << 2
        j = j << 2
        k = k << 2

        r = r + (i & 2**18)
        r = r + (j & 2**19)
        r = r + (k & 2**20)

        i = i << 2
        j = j << 2
        k = k << 2

        r = r + (i & 2**21)
        r = r + (j & 2**22)
        r = r + (k & 2**23)

        i = i << 2
        j = j << 2
        k = k << 2

        r = r + (i & 2**24)
        r = r + (j & 2**25)
        r = r + (k & 2**26)

        i = i << 2
        j = j << 2
        k = k << 2

        r = r + (i & 2**27)
        r = r + (j & 2**28)
        r = r + (k & 2**29)

        i = i << 2
        j = j << 2
        k = k << 2

        r = r + (i & 2**30)
        r = r + (j & 2**31)
        r = r + (k & 2**32)

        return r


    @staticmethod
    @numba.njit(parallel=True)
    def __cub__(r):
        """
        If the binary representation of r is given by:
            bin(r) = (..., k8, j8, i8, k4, j4, i4, k2, j2, i2, k1, j1, i1)
        this function returns the numbers, i, j, and k, for which the binary repressentations are given by:
            bin(i) = (..., i8, i4, i2, i1)
            bin(j) = (..., j8, j4, j2, j1)
            bin(k) = (..., k8, k4, k2, k1)
        inducing an hierarchical ordering.
        """

        i = 0
        j = 0
        k = 0

        # Yes, this can be rewritten as a for-loop
        # Done this way since I am not sure if the compiler unrolls loops.
        # Should be tested!

        i = i + (r & 2**0)
        r = r >> 1
        j = j + (r & 2**0)
        r = r >> 1
        k = k + (r & 2**0)

        i = i + (r & 2**1)
        r = r >> 1
        j = j + (r & 2**1)
        r = r >> 1
        k = k + (r & 2**1)

        i = i + (r & 2**2)
        r = r >> 1
        j = j + (r & 2**2)
        r = r >> 1
        k = k + (r & 2**2)

        i = i + (r & 2**3)
        r = r >> 1
        j = j + (r & 2**3)
        r = r >> 1
        k = k + (r & 2**3)

        i = i + (r & 2**4)
        r = r >> 1
        j = j + (r & 2**4)
        r = r >> 1
        k = k + (r & 2**4)

        i = i + (r & 2**5)
        r = r >> 1
        j = j + (r & 2**5)
        r = r >> 1
        k = k + (r & 2**5)

        i = i + (r & 2**6)
        r = r >> 1
        j = j + (r & 2**6)
        r = r >> 1
        k = k + (r & 2**6)

        i = i + (r & 2**7)
        r = r >> 1
        j = j + (r & 2**7)
        r = r >> 1
        k = k + (r & 2**7)

        i = i + (r & 2**8)
        r = r >> 1
        j = j + (r & 2**8)
        r = r >> 1
        k = k + (r & 2**8)

        i = i + (r & 2**9)
        r = r >> 1
        j = j + (r & 2**9)
        r = r >> 1
        k = k + (r & 2**9)

        i = i + (r & 2**10)
        r = r >> 1
        j = j + (r & 2**10)
        r = r >> 1
        k = k + (r & 2**10)

        i = i + (r & 2**11)
        r = r >> 1
        j = j + (r & 2**11)
        r = r >> 1
        k = k + (r & 2**11)

        i = i + (r & 2**12)
        r = r >> 1
        j = j + (r & 2**12)
        r = r >> 1
        k = k + (r & 2**12)

        i = i + (r & 2**13)
        r = r >> 1
        j = j + (r & 2**13)
        r = r >> 1
        k = k + (r & 2**13)

        i = i + (r & 2**14)
        r = r >> 1
        j = j + (r & 2**14)
        r = r >> 1
        k = k + (r & 2**14)

        i = i + (r & 2**15)
        r = r >> 1
        j = j + (r & 2**15)
        r = r >> 1
        k = k + (r & 2**15)

        return i, j, k
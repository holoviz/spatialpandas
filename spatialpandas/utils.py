from numba import jit

ngjit = jit(nopython=True, nogil=True)

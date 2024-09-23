
from __future__ import print_function

import cython
import numpy as np
cimport numpy as np
from collections import defaultdict



"""
Compiler directives:
https://github.com/cython/cython/wiki/enhancements-compilerdirectives
Cython tutorial:
https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html
Credit to https://github.com/luzai
"""

cdef extern from "math.h":
    float fabs(float num)
        
cdef float[:,:] mp
cdef int n
cdef int[:] link 
cdef double[:] lx
cdef double[:] ly
cdef double[:] sla
cdef int[:] visx
cdef int[:] visy

 

cpdef DFS(int x):
    visx[x] = 1
    for y in range(n):
        if visy[y] == 1:
            continue
        tmp = lx[x] + ly[y] - mp[x][y]
        if fabs(tmp) < 1e-5:
            visy[y] = 1
            if link[y] == -1 or DFS(link[y]):
                link[y] = x
                return True
        elif sla[y] + 1e-5 > tmp: 
            sla[y] = tmp  
    return False

cpdef KM_run(groundMetric):
    global mp, n, link, lx, ly, sla, visx, visy
    mp = groundMetric
    n = groundMetric.shape[0]
    link = np.zeros(n).astype(np.int32)
    lx = np.zeros(n)
    ly = np.zeros(n)
    sla = np.zeros(n)
    visx = np.zeros(n).astype(np.int32)
    visy = np.zeros(n).astype(np.int32)
    
    for index in range(n):
        link[index] = -1
        ly[index] = 0.0
        lx[index] = np.max(mp[index])

    for x in range(n):
        sla = np.zeros(n) + 1e10
        while True:
            visx = np.zeros(n).astype(np.int32)
            visy = np.zeros(n).astype(np.int32)
            if DFS(x): 
                break
            d = 1e10
            for i in range(n):
                if visy[i] == 0:
                    d = min(d, sla[i])
            for i in range(n):
                if visx[i] == 1:
                    lx[i] -= d
                if visy[i] == 1:
                    ly[i] += d
                else:
                    sla[i] -= d

    res = 0
    T = np.zeros((n, n))
    for i in range(n):
        if link[i] != -1:
            T[link[i]][i] = 1.0 / n
    return T

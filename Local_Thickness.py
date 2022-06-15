
"""
Optimized translation of 

Local_Thickness_Parallel.java - Version 3.1
& 
Clean_Up_Local_Thickness.java - Version 3

from the Fiji distribution of ImageJ for the life sciences, 
originally written by Bob Dougherty.

Author: L. L. Longren
"""


import numpy as np
import multiprocessing


class Local_Thickness_Parallel:

    def __init__(self, imp):
        self.stack = imp.copy()
        self.d, self.w, self.h,  = np.shape(self.stack)
        self.very_small_value = 1e-9
        
    def run(self):
        # Create reference to input data
        s = np.empty((self.d, self.w * self.h), dtype=float)
        for k in range(self.d):
            s[k] = self.stack[k].reshape((self.w * self.h))
        # Count the distance ridge points on each slice
        nRidge = np.zeros(self.d, dtype=int)
        print('Local Thickness: scanning stack')
        for k in range(self.d):
            nr = 0
            for j in range(self.h):
                for i in range(self.w):
                    ind = i + self.w * j
                    if s[k][ind] > self.very_small_value:
                        nr += 1
            nRidge[k] = nr
        iRidge, jRidge, rRidge = np.empty((3, self.d), dtype=object)
        # Pull out the distance ridge points
        sMax = 0
        for k in range(self.d):
            nr = nRidge[k]
            iRidge[k] = np.zeros(nr)
            jRidge[k] = np.zeros(nr)
            rRidge[k] = np.zeros(nr)
            sk = s[k]
            iRidgeK = iRidge[k]
            jRidgeK = jRidge[k]
            rRidgeK = rRidge[k]
            iR = 0
            for j in range(self.h):
                for i in range(self.w):
                    ind = i + self.w * j
                    if sk[ind] > self.very_small_value:
                        iRidgeK[iR] = i
                        jRidgeK[iR] = j
                        rRidgeK[iR] = sk[ind]
                        iR += 1
                        if sk[ind] > sMax:
                            sMax = sk[ind]
                        sk[ind] = 0
                        
        nThreads = multiprocessing.cpu_count()
        print('Local Thickness: thread number =', nThreads)
        ltt = self.LTThread(nThreads, self.w, self.h, self.d, nRidge, s, iRidge, jRidge, rRidge, resources=None)
        with multiprocessing.Pool(nThreads) as p:
            sAll = p.map(ltt.run, range(nThreads))

        print('Local Thickness: storing by thread')
        for thread in range(nThreads):
            for k in range(thread, self.d, nThreads):
                s[k] = sAll[thread][k]
            
        # Fix the square values and apply factor of 2
        print('Local Thickness: square root')
        for k in range(self.d):
            for j in range(self.h):
                for i in range(self.w):
                    ind = i + self.w * j
                    s[k][ind] = 2 * np.sqrt(s[k][ind])
        print('Local Thickness complete')
        
        out = np.reshape(s, (self.d, self.w, self.h))
        return out

    class LTThread:
        
        def __init__(self, nThreads, w, h, d, nRidge, s, iRidge, jRidge, rRidge, resources):
            self.nThreads = nThreads
            self.w = w
            self.h = h
            self.d = d
            self.nRidge = nRidge
            self.s = s
            self.iRidge = iRidge
            self.jRidge = jRidge
            self.rRidge = rRidge
            self.resources = resources
            
        def run(self, thread):
            # Loop through ridge points. For each one, update the local thickness for the points within its sphere.
            for k in range(thread, self.d, self.nThreads):
                print('Local Thickness: processing slice {}/{}'.format(k + 1, self.d))
                nR = self.nRidge[k]
                iRidgeK = self.iRidge[k]
                jRidgeK = self.jRidge[k]
                rRidgeK = self.rRidge[k]
                for iR in range(nR):
                    i = int(iRidgeK[iR])
                    j = int(jRidgeK[iR])
                    r = float(rRidgeK[iR])
                    rSquared = int(r * r + 0.5)
                    rInt = int(r)
                    if rInt < r:
                        rInt += 1
                    iStart = i - rInt
                    if iStart < 0:
                        iStart = 0
                    iStop = i + rInt
                    if iStop >= self.w:
                        iStop = self.w - 1
                    jStart = j - rInt
                    if jStart < 0:
                        jStart = 0
                    jStop = j + rInt
                    if jStop >= self.h:
                        jStop = self.h - 1
                    kStart = k - rInt
                    if kStart < 0:
                        kStart = 0
                    kStop = k + rInt
                    if kStop >= self.d:
                        kStop = int(self.d - 1)
                    for k1 in range(kStart, kStop + 1):
                        r1SquaredK = (k1 - k) * (k1 - k)
                        sk1 = self.s[k1]
                        for j1 in range(jStart, jStop + 1):
                            r1SquaredJK = r1SquaredK + (j1 - j) * (j1 - j)
                            if r1SquaredJK <= rSquared:
                                for i1 in range(iStart, iStop + 1):
                                    r1Squared = r1SquaredJK + (i1 - i) * (i1 - i)
                                    if r1Squared <= rSquared:
                                        ind1 = i1 + self.w * j1
                                        s1 = sk1[ind1]
                                        if rSquared > s1:
                                            self.s[k1][ind1] = rSquared
            return self.s
                                            

class Clean_Up_Local_Thickness:

    def __init__(self, imp):
        self.stack = imp.copy()        
        self.d, self.w, self.h,  = np.shape(self.stack)                
        
    def run(self):
        # Create 32 bit floating point stack for output, sNew.             
        sNew = np.empty((self.d, self.w * self.h), dtype=float)   
        # Create reference to input data
        s = np.empty((self.d, self.w * self.h), dtype=float)
        for k in range(self.d):
            s[k] = self.stack[k].reshape((self.w * self.h))    
        print('Clean Up Local Thickness: correcting')   
        # First set the output array to flags:
        # 0 for a background point
        # -1 for a non-background point that borders a background point
        # s (input data) for an interior non-background point
        for k in range(self.d):
            for j in range(self.h):
                for i in range(self.w):
                    sNew[k][i + self.w * j] = self.setFlag(s, i, j, k)
        # Process the surface points. Initially set results to negative values
		# to be able to avoid including them in averages of for subsequent points.
		# During the calculation, positve values in sNew are interior
		# non-background local thicknesses. Negative values are surface points. 
		# In this case the value might be -1 (not processed yet) or -result, where 
		# result is the average of the neighboring interior points. 
		# Negative values are excluded from the averaging.
        for k in range(self.d):
            for j in range(self.h):
                for i in range(self.w):
                    ind = i + self.w * j
                    if sNew[k][ind] == -1:
                        sNew[k][ind] = -self.averageInteriorNeighbors(s, sNew, i, j, k)
        # Fix the negative values and double the results
        for k in range(self.d):
            for j in range(self.h):
                for i in range(self.w):
                    ind = i + self.w * j
                    sNew[k][ind] = abs(sNew[k][ind])
        print('Clean Up Local Thickness complete')
        out = np.reshape(sNew, (self.d, self.w, self.h))
        return out
        
    def setFlag(self, s, i, j, k):
        if s[k][i + self.w * j] == 0: return 0
        # change 1
        if self.look(s, i, j, k - 1) == 0: return -1
        if self.look(s, i, j, k + 1) == 0: return -1
        if self.look(s, i, j - 1, k) == 0: return -1
        if self.look(s, i, j + 1, k) == 0: return -1  
        if self.look(s, i - 1, j, k) == 0: return -1
        if self.look(s, i + 1, j, k) == 0: return -1     
        # change 1 before plus
        if self.look(s, i, j + 1, k - 1) == 0: return -1
        if self.look(s, i, j + 1, k + 1) == 0: return -1
        if self.look(s, i + 1, j - 1, k) == 0: return -1
        if self.look(s, i + 1, j + 1, k) == 0: return -1  
        if self.look(s, i - 1, j, k + 1) == 0: return -1
        if self.look(s, i + 1, j, k + 1) == 0: return -1     
        # change 1 before minus
        if self.look(s, i, j - 1, k - 1) == 0: return -1
        if self.look(s, i, j - 1, k + 1) == 0: return -1
        if self.look(s, i - 1, j - 1, k) == 0: return -1
        if self.look(s, i - 1, j + 1, k) == 0: return -1  
        if self.look(s, i - 1, j, k - 1) == 0: return -1
        if self.look(s, i + 1, j, k - 1) == 0: return -1     
        # change 3, k+1
        if self.look(s, i + 1, j + 1, k + 1) == 0: return -1
        if self.look(s, i + 1, j - 1, k + 1) == 0: return -1
        if self.look(s, i - 1, j + 1, k + 1) == 0: return -1
        if self.look(s, i - 1, j - 1, k + 1) == 0: return -1
        # change 3, k-1
        if self.look(s, i + 1, j + 1, k - 1) == 0: return -1
        if self.look(s, i + 1, j - 1, k - 1) == 0: return -1
        if self.look(s, i - 1, j + 1, k - 1) == 0: return -1
        if self.look(s, i - 1, j - 1, k - 1) == 0: return -1
        return s[k][i + self.w * j]
        
    def averageInteriorNeighbors(self, s, sNew, i, j, k):
        n: int = 0
        total: float = 0  # (sum)
        # change 1
        value: float = self.lookNew(sNew, i, j, k - 1)
        if value > 0:
            n += 1
            total += value
        value: float = self.lookNew(sNew, i, j, k + 1)
        if value > 0:
            n += 1
            total += value
        value: float = self.lookNew(sNew, i, j - 1, k)
        if value > 0:
            n += 1
            total += value
        value: float = self.lookNew(sNew, i, j + 1, k)
        if value > 0:
            n += 1
            total += value
        value: float = self.lookNew(sNew, i - 1, j, k)
        if value > 0:
            n += 1
            total += value
        value: float = self.lookNew(sNew, i + 1, j, k)
        if value > 0:
            n += 1
            total += value
        # change 1 before plus
        value: float = self.lookNew(sNew, i, j + 1, k - 1)
        if value > 0:
            n += 1
            total += value
        value: float = self.lookNew(sNew, i, j + 1, k + 1)
        if value > 0:
            n += 1
            total += value
        value: float = self.lookNew(sNew, i + 1, j - 1, k)
        if value > 0:
            n += 1
            total += value
        value: float = self.lookNew(sNew, i + 1, j + 1, k)
        if value > 0:
            n += 1
            total += value
        value: float = self.lookNew(sNew, i - 1, j, k + 1)
        if value > 0:
            n += 1
            total += value
        value: float = self.lookNew(sNew, i + 1, j, k + 1)
        if value > 0:
            n += 1
            total += value
        # change 1 before minus
        value: float = self.lookNew(sNew, i, j - 1, k - 1)
        if value > 0:
            n += 1
            total += value
        value: float = self.lookNew(sNew, i, j - 1, k + 1)
        if value > 0:
            n += 1
            total += value
        value: float = self.lookNew(sNew, i - 1, j - 1, k)
        if value > 0:
            n += 1
            total += value
        value: float = self.lookNew(sNew, i - 1, j + 1, k)
        if value > 0:
            n += 1
            total += value
        value: float = self.lookNew(sNew, i - 1, j, k - 1)
        if value > 0:
            n += 1
            total += value
        value: float = self.lookNew(sNew, i + 1, j, k - 1)
        if value > 0:
            n += 1
            total += value
        # change 3, k+1
        value: float = self.lookNew(sNew, i + 1, j + 1, k + 1)
        if value > 0:
            n += 1
            total += value
        value: float = self.lookNew(sNew, i + 1, j - 1, k + 1)
        if value > 0:
            n += 1
            total += value
        value: float = self.lookNew(sNew, i - 1, j + 1, k + 1)
        if value > 0:
            n += 1
            total += value
        value: float = self.lookNew(sNew, i - 1, j - 1, k + 1)
        if value > 0:
            n += 1
            total += value
        # change 3, k-1
        value: float = self.lookNew(sNew, i + 1, j + 1, k - 1)
        if value > 0:
            n += 1
            total += value
        value: float = self.lookNew(sNew, i + 1, j - 1, k - 1)
        if value > 0:
            n += 1
            total += value
        value: float = self.lookNew(sNew, i - 1, j + 1, k - 1)
        if value > 0:
            n += 1
            total += value
        value: float = self.lookNew(sNew, i - 1, j - 1, k - 1)
        if value > 0:
            n += 1
            total += value
        if n > 0:
            return total / n
        return s[k][i + self.w * j]
    
    def look(self, s, i, j, k):
        if (i < 0) or (i >= self.w): return -1
        if (j < 0) or (j >= self.h): return -1
        if (k < 0) or (k >= self.d): return -1
        return s[k][i + self.w * j]
    
    # A positive result means this is an interior, non-background, point.    
    def lookNew(self, sNew, i, j, k):
        if (i < 0) or (i >= self.w): return -1
        if (j < 0) or (j >= self.h): return -1
        if (k < 0) or (k >= self.d): return -1
        return sNew[k][i + self.w * j]
        
                                 
        

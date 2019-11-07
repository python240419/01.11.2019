# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 00:43:31 2019

@author: amiha
"""
import numpy as np
import matplotlib.pyplot as plt

class Matrix:
    """
    Represents a rectangular matrix with n rows and m columns.
    """

    def __init__(self, n, m, val=0):
        """
        Create an n-by-m matrix of val's.
        Inner representation: list of lists (rows)
        """
        assert n > 0 and m > 0
        #self.rows = [[val]*m]*n #why this is bad?
        self.rows = [[val]*m for i in range(n)]

    def dim(self):
        return len(self.rows), len(self.rows[0])

    def __repr__(self):
        if len(self.rows)*len(self.rows[0])>1000:
            return "Matrix too large, specify submatrix"
        return "<Matrix {}>".format(self.rows)

    def __eq__(self, other):
        return isinstance(other, Matrix) and self.rows == other.rows

    def copy(self):
        ''' brand new copy of matrix '''
        n,m = self.dim()
        new = Matrix(n,m)
        for i in range (n):
            for j in range (m):
                new[i,j] = self[i,j]
        return new

    # cell/sub-matrix access/assignment
    ####################################
    #ij is a tuple (i,j). Allows m[i,j] instead m[i][j]
    def __getitem__(self, ij): 
        i,j = ij
        if isinstance(i, int) and isinstance(j, int):
            return self.rows[i][j]
        elif isinstance(i, slice) and isinstance(j, slice):
            M = Matrix(1,1) # to be overwritten
            M.rows = [row[j] for row in self.rows[i]]
            return M
        else:
            return NotImplemented

    #ij is a tuple (i,j). Allows m[i,j] instead m[i][j]    
    def __setitem__(self, ij, val): 
        i,j = ij
        if isinstance(i,int) and isinstance(j,int):
            assert isinstance(val, (int, float, complex))
            self.rows[i][j] = val
        elif isinstance(i,slice) and isinstance(j,slice):
            assert isinstance(val, Matrix)
            n,m = val.dim()
            s_rows = self.rows[i]
            assert len(s_rows) == n and len(s_rows[0][j]) == m
            for s_row, v_row in zip(s_rows,val.rows):
                s_row[j] = v_row
        else:
            return NotImplemented

    # arithmetic operations
    ########################
    def entrywise_op(self, other, op):
        if not isinstance(other, Matrix):
            return NotImplemented
        assert self.dim() == other.dim()
        n,m = self.dim()
        M = Matrix(n,m)
        for i in range(n):
            for j in range(m):
                M[i,j] = op(self[i,j], other[i,j])
        return M

    def __add__(self, other):
        return self.entrywise_op(other,lambda x,y:x+y)


    def __sub__(self, other):
        return self.entrywise_op(other,lambda x,y:x-y)
    
    def __neg__(self):
        n,m = self.dim()
        return Matrix(n,m) - self


    def __mul__(self, other):
        if isinstance(other, Matrix):
            return self.multiply_by_matrix(other)
        elif isinstance(other, (int, float, complex)):
            return self.multiply_by_scalar(other)
        else:
            return NotImplemented

    __rmul__ = __mul__
        
    def multiply_by_scalar(self, val):
        n,m = self.dim()
        return self.entrywise_op(Matrix(n,m,val), lambda x,y :x*y)
###a more efficient version, memory-wise. 
##        n,m = self.dim()
##        M = Matrix(n,m)
##        for i in range(n):
##            for j in range(m):
##                M[i,j] = self[i,j] * val
##        return M

    def multiply_by_matrix(self, other):
        assert isinstance(other, Matrix)
        n,m = self.dim()
        n2,m2 = other.dim()
        assert m == n2
        M = Matrix(n,m2)
        for i in range(n):
            for j in range(m2):
                M[i,j] = sum(self[i,k] * other[k,j] for k in range(m))
        return M


    # Input/output
    ###############
    def save(self, filename):
        f = open(filename, 'w')
        n,m = self.dim()
        print(n,m, file=f)
        for row in self.rows:
            for e in row:
                print(e, end=" ", file=f)
            print("",file=f) #newline
        f.close()

    @staticmethod
    def load(filename):
        f = open(filename)
        line = f.readline()
        n,m = [int(x) for x in line.split()]
        result = Matrix(n,m)
        for i in range(n):
            line = f.readline()
            row = [int(x) for x in line.split()]
            assert len(row) == m
            result.rows[i] = row
        return result

    # display - for image visualization - using plt
    ###############################################
    def display(self, title=None, zoom=None):
        X = np.array(self.rows)
        plt.imshow(X, cmap="gist_gray" )
        # plt.clim(0,100)
        n,m = self.dim()
        plt.figure(figsize=(1+n/100,1+m/100))
        plt.show( )

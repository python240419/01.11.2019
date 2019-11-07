#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np

############################################################################
###    class Matrix
###    This is our own Matrix class with an additional display() method for image visualization
###    Also, functions 'save' and 'load' enable working with .bitmap files
###    Note that __repr__ avoids printing very large matrices (would stuck IDLE)
############################################################################

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


# In[3]:


def join_h(mat1, mat2):
    """ joins two matrices, side by side with some separation """
    n1,m1 = mat1.dim()
    n2,m2 = mat2.dim()
    m = m1+m2+10
    n = max(n1,n2)
    new = Matrix(n, m, val=255)  # fill new matrix with white pixels

    new[:n1,:m1] = mat1
    new[:n2,m1+10:m] = mat2

    return new

def join_v(mat1, mat2):
    """ joins two matrices, vertically with some separation """
    n1,m1 = mat1.dim()
    n2,m2 = mat2.dim()
    n = n1+n2+10
    m = max(m1,m2)
    new = Matrix(n, m, val=255)  # fill new matrix with white pixels

    new[:n1,:m1] = mat1
    new[n1+10:n,:m2] = mat2

    return new

def join(*mats, direction):
    ''' *mats enables a variable number of parameters.
        direction is either 'h' or 'v', for horizontal 
        or vertical join, respectively '''
    func = join_v if direction == 'v' else join_h
    res = mats[0] #first matrix parameter
    for mat in mats[1:]:
        res = func(res, mat)
    return res


# In[2]:


abbey=Matrix.load('abbey_road.bitmap')
abbey.display()
abbey[-3:-1,-100:-1]


# In[4]:


einstein=Matrix.load('albert-einstein-1951.bitmap')
einstein.display()


# In[5]:


NotImplemented


# In[6]:


import random
lst=[random.gauss(0,10) for i in range(20)]
print(lst)


# In[7]:


print(sorted(lst))


# In[18]:


##################################
## Adding noise to images,
## for testing noise reduction
##################################

def add_gauss(mat, sigma=10):
    ''' Generates Gaussian noise with mean 0 and SD sigma.
        Adds indep. noise to pixel,
        keeping values in 0..255'''
    n,m = mat.dim()
    new = mat.copy()
    for i in range(n):
        for j in range(m):
            noise = round(random.gauss(0,sigma))
            if noise > 0:
                new[i,j] = min(mat[i,j] + noise, 255)
            elif noise < 0:
                new[i,j] = max(mat[i,j] + noise, 0)

    return new


# In[19]:


new10 = add_gauss (abbey)
new20 = add_gauss (abbey, sigma =20)
new50 = add_gauss (abbey, sigma =50)
joinedAbbeyGauss = join( new10, new20, new50, direction='h' )
joinedAbbeyGauss.display()


# In[8]:


##################################
## Local denoising methods
##################################

def items(mat):
    ''' flatten mat elements into a list '''
    n,m = mat.dim()
    lst = [mat[i,j] for i in range(n) for j in range(m)]
    return lst


def local_operator(mat, op, k=1):
    ''' Apply op to every pixel.
        op is a local operator on a square neighbourhood
        of size 2k+1 X 2k+1 around a pixel '''
    n,m = mat.dim()
    res = mat.copy()  # brand new copy of A
    for i in range(k,n-k):
        for j in range(k,m-k):
            res[i,j] = op(items(mat[i-k:i+k+1,j-k:j+k+1]))
    return res


def average(lst):
    n = len(lst)
    return round(sum(lst)/n)

def local_means(mat, k=1):
    return local_operator(mat, average, k)


def median(lst):
    sort_lst = sorted(lst)
    n = len(sort_lst)
    if n%2==1:    # odd number of elements. well defined median
        return sort_lst[n//2]
    else:         # even number of elements. average of middle two
        return (int(sort_lst[-1+n//2]) + int(sort_lst[n//2])) // 2


def local_medians(mat, k=1):
    return local_operator(mat, median, k)


# In[21]:


mat = Matrix (4 ,4)
for i in range (4):
    for j in range (4):
        mat[i,j] = i + (j**2)
for i in range (4):
    print ([mat[i,j] for j in range (4)])
#Note that 12 gets converted to White
# by plt
mat.display()    


# In[22]:


mat2 = local_means (mat)
for i in range (4):
    print ([mat2[i,j] for j in range (4)])
mat2.display()


# In[24]:


mat[2,2] = 255
for i in range (4):
    print ([mat[i,j] for j in range (4)])
mat.display()


# In[66]:


mat2 = local_means (mat)
for i in range (4):
    print ([mat2[i,j] for j in range (4)])
mat2.display()    


# In[25]:


def add_SP(mat, p=0.01):
    ''' Generates salt and pepper noise:
        Each pixel is "hit" indep. with prob. p
        If hit, it has fifty fifty chance of becoming
        white or black. '''
    n,m = mat.dim()
    new = mat.copy()
    for i in range(n):
        for j in range (m):
            rand = random.random() #a random float in [0,1)
            if rand < p:
                if rand < p/2:
                    new[i,j] = 0
                else:
                    new[i,j] = 255
    return new


# In[26]:


sp1 = add_SP(abbey)
sp2 = add_SP(abbey, p =0.02)
sp5 = add_SP(abbey, p =0.05)
joined1 = join(abbey, sp1, direction='h')
joined2 = join(sp2, sp5, direction='h')
joinedAbbeySP = join(joined1, joined2, direction = 'v')
joinedAbbeySP.display()


# In[27]:


mat = Matrix (4 ,4)
for i in range (4):
    for j in range (4):
         mat[i,j] = i + (j**2)
for i in range (4):
    print ([mat[i,j] for j in range (4)])
mat.display()


# In[28]:


matb = local_medians(mat)
for i in range (4):
    print ([matb[i,j] for j in range (4)])
matb.display()


# In[29]:


mat = Matrix (4 ,4)
mat[2,2]=255
print("Original:")
for i in range (4):
    print ([mat[i,j] for j in range (4)])
matb = local_means(mat)
print("\nLocal Means:")
for i in range (4):
    print ([matb[i,j] for j in range (4)])
matc = local_medians(mat)
print("\n Local Medians:")
for i in range (4):
    print ([matc[i,j] for j in range (4)])


# In[30]:


#this code will take a few seconds to run
denoised_by_means = local_means(joinedAbbeySP)
denoised_by_means.display()


# In[31]:


#this code will take a few seconds to run
denoised_by_medians = local_medians(joinedAbbeySP)
denoised_by_medians.display()


# In[32]:


#this code will take a few seconds to run
Gauss_denoised_by_means = local_means(joinedAbbeyGauss)
Gauss_denoised_by_means.display()


# In[33]:


#this code will take a few seconds to run
Gauss_denoised_by_medians = local_medians(joinedAbbeyGauss)
Gauss_denoised_by_medians.display()


# In[34]:


def segment(mat, threshold):
    ''' Binary segmentation of image (matrix) 
        using a threshold
    '''
    n,m = mat.dim()
    out = Matrix(n,m)
    
    for x in range(n):
        for y in range(m):
            if mat[x, y] >= threshold:
                out[x,y] = 255 #white
            else:
                out[x,y] = 0 #black

    return out


# In[35]:


einstein.display()
segment(einstein, 50).display()
segment(einstein, 100).display()
segment(einstein, 150).display()


# In[36]:


segment(abbey, 50).display()
segment(abbey, 100).display()
segment(abbey, 150).display()


# In[37]:


def histogram(mat):
    ''' Return a histogram as a list,
        where index i hold the number of pixels with value I
    '''

    width, height = mat.dim()
    hist = [0]*256
    
    for x in range(width):
        for y in range(height):
            gray_level= mat[x,y]
            hist[gray_level] += 1
  
    return hist #hist[i] = number of pixels with gray level=i


# In[38]:


einstein=Matrix.load('albert-einstein-1951.bitmap')
plt.hist( getValues(einstein.rows), bins=256 )
print()


# In[ ]:


segment(einstein, 45).display()


# In[ ]:


segment(einstein, 110).display()


# # Applying K-Means to cluster image colors

# In[ ]:


from sklearn.cluster import KMeans
from copy import deepcopy

def getValues(r):
    arr = np.array(r)
    rows,cols = arr.shape
    return arr.reshape(rows*cols), rows, cols

def clusterImageColors(mat, num):
    print("Clustering to", num, "clusters...")
    # Number of clusters = number of colors
    K = num

    #list of lists ==> 1d np array
    data,rows,cols = getValues(mat.rows)
    n = len(data)

    kmeans = KMeans(n_clusters=K, random_state=0).fit( data.reshape(-1,1) )

    # Returns K vectors that represent the cluster centers
    centroids = kmeans.cluster_centers_.astype(int)

    # Assigns each column (student) to a centroid (->cluster)
    assignments = kmeans.labels_

    for ix,val in enumerate(data):
        data[ix] = centroids[assignments[ix]]

    # 1d array ==> 2d array (original structure)
    data = data.reshape(rows,cols).astype(int)    

    matKMeans = mat.copy()
    for ix_r,row in enumerate(data):
        for ix_c,col in enumerate(data[ix_r]):
            #convert to "regular" int
            matKMeans[ix_r,ix_c] = np.asscalar(np.int16(data[ix_r,ix_c]))
    return matKMeans


abbey = Matrix.load('abbey_road.bitmap')            
abbey.display()
abbey2Colors = clusterImageColors(abbey, 2)
abbey2Colors.display()
abbey3Colors = clusterImageColors(abbey, 3)
abbey3Colors.display()


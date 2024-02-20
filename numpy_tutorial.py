# link to the tutorial: https://www.youtube.com/watch?v=QUT1VHiLmmI
# numpy is multidimensional array library
# numpy is faster than python list, it's faster to read less bytes of memory
# SIMD Vecotr Processing: Single Instruction, Multiple Data
# you can multiply two arrays together and numpy will
# do the multiplication element by element

import numpy as np

# The Basics
a = np.array([1, 2, 3])
b = np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]])

# get dimension
a.ndim

# get shape
b.shape

# get type
a.dtype

# get size
a.itemsize

# get total size
a.size * a.itemsize
# or
a.nbytes


# Accessing/Changing specific elements, rows, columns, etc
a = np.array([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])

# get a specific element [r, c]
a[1, 5]  # 13

# get a specific row
a[0, :]  # [1, 2, 3, 4, 5, 6, 7]

# get a specific column
a[:, 2]  # [3, 10]


# getting a little more fancy [startindex:endindex:stepsize]
a[0, 1:6:2]  # [2, 4, 6]

# change an element
a[1, 5] = 20

# change a column
a[:, 2] = [1, 2]

# 3-d example
b = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# get specific element (work outside in)
b[0, 1, 1]  # 4

# replace
b[:, 1, :] = [[9, 9], [8, 8]]


# Initializing Different Types of Arrays

# all 0s matrix
np.zeros((2, 3))  # 2 rows, 3 columns

# all 1s matrix
np.ones((4, 2, 2))  # 4 matrices, 2 rows, 2 columns

# any other number
np.full((2, 2), 99)  # 2x2 matrix with 99s

# any other number (full_like)
np.full_like(a, 4)  # 2x7 matrix with 4s

# random decimal numbers
np.random.rand(4, 2)  # 4x2 matrix with random numbers

# random integer values
np.random.randint(7, size=(3, 3))  # 3x3 matrix with random numbers from 0 to 6

# the identity matrix
np.identity(5)  # 5x5 matrix with 1s on the diagonal

# repeat an array
arr = np.array([[1, 2, 3]])
r1 = np.repeat(arr, 3, axis=0)  # repeat arr 3 times along the rows


# excerise 1
output = np.ones((5, 5))
z = np.zeros((3, 3))
z[1, 1] = 9
output[1:4, 1:4] = z


# be careful when copying arrays
a = np.array([1, 2, 3])
b = a.copy()  # not b = a because it will change a as well


# Mathematics
a = np.array([1, 2, 3, 4])
a + 2  # [3, 4, 5, 6]
a - 2  # [-1, 0, 1, 2]
a * 2  # [2, 4, 6, 8]
a / 2  # [0.5, 1.0, 1.5, 2.0]
b = np.array([1, 0, 1, 0])
a + b  # [2, 2, 4, 4]
a ** 2  # [1, 4, 9, 16]
np.sin(a)  # [sin(1), sin(2), sin(3), sin(4)]
np.cos(a)  # [cos(1), cos(2), cos(3), cos(4)]


# Linear Algebra
a = np.ones((2, 3))
b = np.full((3, 2), 2)

# matrix multiplication
np.matmul(a, b)

# find the determinant
c = np.identity(3)
np.linalg.det(c)  # 1.0


# Statistics
stats = np.array([[1, 2, 3], [4, 5, 6]])
np.min(stats)  # 1
np.max(stats)  # 6
np.min(stats, axis=1)  # [1, 4]
np.max(stats, axis=0)  # [4, 5, 6]
np.sum(stats)  # 21


# Reorganizing Arrays
before = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
after = before.reshape((8, 1))  # 8 rows, 1 column
after = before.reshape((2, 2, 2))  # 2 matrices, 2 rows, 2 columns

# vertically stacking vectors
v1 = np.array([1, 2, 3, 4])
v2 = np.array([5, 6, 7, 8])
np.vstack([v1, v2])  # [[1, 2, 3, 4], [5, 6, 7, 8]]

# horizontal stack
h1 = np.ones((2, 4))
h2 = np.zeros((2, 2))
np.hstack((h1, h2))  # [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0]]


# Miscellaneous

# load data from file
filedata = np.genfromtxt('data.txt', delimiter=',')
filedata = filedata.astype('int32')

# boolean masking and advanced indexing
filedata > 50  # boolean mask of the array
filedata[filedata > 50]  # array of the values that are greater than 50
# you can index with a list in numpy
np.any(filedata > 50, axis=0)  # check if any value in each column is > 50
(filedata > 50) & (filedata < 100)

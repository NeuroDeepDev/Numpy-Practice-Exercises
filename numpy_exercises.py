# Exercise 1. Import numpy as np and see the version
import numpy as np
print(np.__version__)

# Exercise 2. Create a 1D array
arr = np.array([1,2,3,4,5])
print(arr)

# Exercise 3. Create a boolean array
arr1 = np.full((2,2), True, dtype=bool)
print(arr1)

# Exercise 4. Extract items based on condition
arr2 = np.array([1,2,3,5,6,7,8])
arr2[arr2 % 2 == 1]

# Exercise 5. Replace items based on condition
arr2[arr2 % 2 == 1] = 0
arr2

# Exercise 6. Replace items without affecting original array
ar = np.arange(10)
ari = ar.copy()
ari[ari % 2 == 0] = 0
print(ari)
print(ar)

# Exercise 7. Reshape an array
print(ar.reshape(2,5))
print(ar.shape)

# Exercise 8. Stack two arrays vertically
a = np.array([1,2,3,5,3,2,4,5]).reshape(2,4)
b = np.full((2,4), range(2,6), dtype=int)
np.vstack([a,b])

# Exercise 9. Stack two arrays horizontally
np.hstack([a,b])

# Exercise 10. Generate custom sequences
l1 = np.array([1,2,3])
np.hstack([np.repeat(l1, 3), np.tile(l1, 3)])

#Exercise 11. How to get the common items between two python numpy arrays?
np.intersect1d(a,b)

#Exercise 12. How to remove from one array those items that exist in another?
np.setdiff1d(a,b)

#Exercise 13. How to get the positions where elements of two arrays match?
np.where(a==b)

#Exercise 14. How to extract all numbers between a given range from a numpy array?
l1 = np.arange(15)
l1[(l1 > 2) & (l1 < 14)]

#Exercise 15. How to make a python function that handles scalars to work on numpy arrays?
def circumference(r):
  return 3.14 * r * 2
print(circumference(l1))
vec_cir = np.vectorize(circumference)
vec_cir(l1)

# Exercise 16. How to swap two columns in a 2d numpy array?
arr = np.arange(6).reshape(2,3)
print("Before modification")
print(arr)
print("After modefication")
arr[:, [2,0,1]]

#Exercise 17. How to swap two rows in a 2d numpy array?
arrr = np.arange(6).reshape(3,2)
print("Before modification")
print(arrr)
print("After modefication")
arrr[[1,0,2], :]

#Exercise 18. How to reverse the rows of a 2D array?
arrr[::-1, :]

# Exercise 19. How to reverse the columns of a 2D array?
print(arr)
arr[:, ::-1]

# Exercise 20. How to create a 2D array containing random floats between 5 and 10?
arr_rand = np.random.uniform(5,10, size = (3,5))
arr_rand

#Exercise 21. How to print only 3 decimal places in python numpy array?
arr_rand_rand = np.random.random((3,5))
arand = np.round(arr_rand_rand, 3)
arand

# Exercise 22. How to pretty print a numpy array by suppressing the scientific notation (like 1e10)?
np.random.seed(100)
rand_arr = np.random.random([3,3])/1e3
np.set_printoptions(suppress = False)
rand_arr
np.set_printoptions(suppress = True)
rand_arr

# Exercise 23. How to limit the number of items printed in output of numpy array?
z = np.arange(20)
np.set_printoptions(threshold = 1)
z

# Exercise 24. How to print the full numpy array without truncating
np.set_printoptions(threshold = 20)
z

# Exercise 25. How to import a dataset with numbers and texts keeping the text intact in python numpy?
url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
data = np.genfromtxt(url, delimiter = ',', dtype = None, encoding='utf-8')
dataata[:5]

# Exercise 26. How to extract a particular column from 1D array of tuples?¶
data[:5]

# Exercise 27. How to convert a 1d array of tuples to a 2d numpy array?
iris_data = np.genfromtxt('../input/iris/Iris.csv', delimiter=',', skip_header=1, dtype='float', usecols=[0,1,2,3])
iris_data

# Exercise 28. How to compute the mean, median, standard deviation of a numpy array?
iris_data = np.genfromtxt('../input/iris/Iris.csv', delimiter=',', skip_header=1, usecols = [1])
print('Mean', np.mean(iris_data))
print('Median', np.median(iris_data))
print('Standard Deviation', np.std(iris_data))

# Exercise 29. How to normalize an array so the values range exactly between 0 and 1?
iris_data = np.genfromtxt('../input/iris/Iris.csv', delimiter=',', dtype='float', usecols=[1], skip_header=1)
(iris_data - np.min(iris_data))/(np.max(iris_data) - np.min(iris_data))

# Exercise 30. How to compute the softmax score?
iris_data = np.genfromtxt('../input/iris/Iris.csv', delimiter=',', dtype='float', usecols=[1], skip_header=1)
softmax = np.exp(iris_data)/sum(np.exp(iris_data))
softmax.sum() # it must sum 1
softmax

# Exercise 31. How to find the percentile scores of a numpy array?
id = np.genfromtxt('/kaggle/input/iris/Iris.csv', delimiter = ',', usecols = [1], dtype = float, skip_header = 1)
np.percentile(id, [5, 95])

# Exercise 32. How to insert values at random positions in an array?
id = np.genfromtxt('/kaggle/input/iris/Iris.csv', delimiter = ',', usecols = [1,2,3,4], dtype = 'float',skip_header = 1)
for i in np.random.randint(1, len(id), 20):
    id[i] = np.nan
id[:5]

# Exercise 33. How to find the position of missing values in numpy array?
id = np.genfromtxt('/kaggle/input/iris/Iris.csv', delimiter = ',', usecols = [1,2,3,4], dtype = 'float',skip_header = 1)

id[ # replacing selected value to nan
np.random.randint(len(id), size = 20),
np.random.randint(4, size = 20)
] = np.nan

# Find total mising value in complete data
print(f"Number of missing values:{np.isnan(id[:, :]).sum()}")

# Find total mising value in 1D data
print(f"Number of missing values in any one feature of Iris data:{np.isnan(id[:, 0]).sum()}")0]))}")

# Exercise 34. How to filter a numpy array based on two or more conditions?
id = np.genfromtxt('/kaggle/input/iris/Iris.csv', delimiter = ',', usecols = [1,2,3,4], dtype = 'float', skip_header = 1)
a = id[(id[:, 0] > 0.5 ) & (id[:, 1] > 2.5 )]
a[:5]

# Exercise 35. How to drop rows that contain a missing value from a numpy array?
id = np.genfromtxt('/kaggle/input/iris/Iris.csv', delimiter = ',', usecols = [1,2,3,4], dtype = 'float',skip_header = 1)

id[ # replacing selected value to nan
np.random.randint(len(id), size = 20),
np.random.randint(4, size = 20)
] = np.nan

id[np.sum(np.isnan(id,), axis = 1)][:5]

# Exercise 36. How to find the correlation between two columns of a numpy array?
id = np.genfromtxt('/kaggle/input/iris/Iris.csv', delimiter = ',', usecols = [1,2,3,4], dtype = 'float',skip_header = 1)
print(np.corrcoef(id[:1], id[:5]))
print()
np.corrcoef(id[:1], id[:5])[0,1]

# Exercise 37. How to find if a given array has any null values?
id = np.genfromtxt('/kaggle/input/iris/Iris.csv', delimiter = ',', usecols = [1,2,3,4], dtype = 'float',skip_header = 1)
np.isnan(id).any()
# Exercise 38. How to replace all missing values with 0 in a numpy array?
id = np.genfromtxt('/kaggle/input/iris/Iris.csv', delimiter = ',', usecols = [1,2,3,4], dtype = 'float',skip_header = 1)

id[ # replacing selected value to nan
np.random.randint(len(id), size = 20),
np.random.randint(4, size = 20)
] = np.nan

print("Does dataset have any Nan value:",np.isnan(id).any())
id[np.isnan(id)] = 0
print("Does dataset have any Nan value:",np.isnan(id).any())

# Exercise 39. How to find the count of unique values in a numpy array?
id = np.genfromtxt('/kaggle/input/iris/Iris.csv', delimiter = ',', usecols = [1,2,3,4], dtype = 'float',skip_header = 1)

np.unique(id, return_counts = True)

# Exercise 40. How to convert a numeric to a categorical (text) array?
id = np.genfromtxt('/kaggle/input/iris/Iris.csv', delimiter = ',', usecols = [3], dtype = object, skip_header = 1)

bins = np.array([0, 3, 5, 7])
inds = np.digitize(id.astype('float'), bins)

labels = {1:'small', 2: 'medium', 3:'large'}
icd = [labels[x] for x in inds]
icd[:10]

# Exercise 41. How to create a new column from existing columns of a numpy array?
id = np.genfromtxt('../input/iris/Iris.csv', delimiter = ',', usecols = [1,2,3,4], dtype = 'float',skip_header = 1)
new_col = id[:, 0] + id[:, 2]
result = np.column_stack((id, new_col))
result[:5]

# Exercise 42. How to do probabilistic sampling in numpy?
a = np.array(['Deepanshu', 'Ansh', 'kishor', 'chirag', 'Durgesh'])
prob = [0.2, 0.1, 0.2, 0.2, 0.3]
species_out = np.random.choice(a, size = 1, p= prob)
species_out

# Exercise 43. How to get the second largest value of an array when grouped by another array?
a = np.array([15,20,30,40,50,60])
b = np.array(['A','A','B','B','A','B'])

result = {}

for group in np.unique(b):
    vals = a[b == group]
    result[group] = np.sort(vals)[-2]
result

# Exercise 44. How to sort a 2D array by a column
id = np.genfromtxt('../input/iris/Iris.csv', delimiter = ',', usecols = [1,2,3,4], dtype = 'float',skip_header = 1)
id[id[:, 2].argsort()]

# Exercise 45. How to find the most frequent value in a numpy array?
id = np.genfromtxt('../input/iris/Iris.csv', delimiter = ',', dtype = object, usecols = [1,2,3,4,5], skip_header = 1)
v,c = np.unique(id[:, 2], return_counts = True)
v[np.max(c)]

# Exercise 46. How to find the position of the first occurrence of a value greater than a given value?
id = np.genfromtxt('../input/iris/Iris.csv', delimiter = ',', dtype = object, usecols = [4], skip_header = 1)
np.argwhere(id[:].astype(float) > 0.4)[0]

# Exercise 47. How to replace all values greater than a given value to a given cutoff?
id = np.genfromtxt('../input/iris/Iris.csv',
                   delimiter=',', dtype=float,
                   usecols=[2], skip_header=1)

# replace values greater than 3.0 with 3.0
id[id > 3.5] = 1010

print(id[:10])

# Exercise 48. How to get the positions of top n values from a numpy array?
id = np.genfromtxt('../input/iris/Iris.csv',
                   delimiter=',', dtype=float,
                   usecols=[2], skip_header=1)

top5_idx = np.argsort(id)[-5:]

print(top5_idx)


# Exercise 49. How to compute the row wise counts of all possible values in an array?
a = np.array([[1, 2, 2, 3],
              [2, 2, 3, 3],
              [1, 1, 1, 2]])

result = []

for row in a:
    values, counts = np.unique(row, return_counts=True)
    result.append(dict(zip(values, counts)))

print(result)

# Exercise 50. How to convert an array of arrays into a flat 1d array?
a = np.arange(4)
b = np.arange(4,7)
c = np.arange(7,11)
np.concatenate([a,b,c])

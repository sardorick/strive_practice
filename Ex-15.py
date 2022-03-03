#### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆) 
(**hint**: array\[1:-1, 1:-1\])
# arr = np.ones((5,5))
# arr[1:-1, 1:-1] = 0
# print(arr)

arr = np.zeros((5,5))
arr = np.pad(arr, pad_width=1, mode='constant', constant_values=1)
print(arr)
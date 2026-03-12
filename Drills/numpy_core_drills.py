import numpy as np  
arr_a = np.array('43')
print(arr_a)
print(type(arr_a))
print(arr_a.shape)  
print(arr_a.ndim)
arr_b = np.array(['1','2','3'])
print(arr_b)
print(type(arr_b))
print(arr_b.shape)  
print(arr_b.ndim)

arr_c = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr_c         )
print(type(arr_c))
print(arr_c.shape)  
print(arr_c.ndim)
arr_d = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]]])
print(arr_d)
print(type(arr_d))
print(arr_d.shape)  
print(arr_d.ndim)
arr_e = np.array([1,2,3],ndmin= 6)
print(arr_e)
print(type(arr_e))
print(arr_e.shape)  
print(arr_e.ndim)

range_array = np.arange(0,100)
print(range_array)

random = np.random.randint(1,100,size=10)
print(random)

X = np.random.randn(100,1)
print(X)

y = 4+3*X+ np.random.randn(100,1)
print(y)

print(X.T)
print(X@X.T)
print(X.mean())
print(X.std())
print(X.var())
print(X.sum())
print(X.min())
print(X.max())
print(X.dot(X.T))


X = np.round(np.random.randn(100,1),3)
print("X: ", X)

y = 4+(3*X)+ np.round(np.random.randn(100,1),3)
print("Y: ",y)

X_b = np.c_[np.ones((100,1)), X] 
print("X_b: ",X_b)


theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print("Theta :" ,theta)



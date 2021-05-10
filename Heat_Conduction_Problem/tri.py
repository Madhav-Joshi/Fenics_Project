import numpy as np
import matplotlib.pyplot as plt
output = open('./Heat_Conduction_Problem/heat_cond_1','rb')
npzfile = np.load(output)
print(npzfile.files)
T = npzfile['vertex_values_T']
x = npzfile['x']
y = npzfile['y']
z = npzfile['z']

x1 = x[T>727]
y1 = y[T>727]
z1 = z[T>727]

T1 = T[T>727]
print(z1)
print(np.max(T1))
'''fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(x1,y1,z1)
plt.show()'''
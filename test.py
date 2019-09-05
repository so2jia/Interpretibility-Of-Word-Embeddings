import numpy as np

a = np.array([[2, 7, 3],
              [3, 5, 1]])

print(a)

a = np.argsort(a[0, :])
print(a)
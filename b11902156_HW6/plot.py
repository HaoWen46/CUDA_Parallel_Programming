import numpy as np
import matplotlib.pyplot as plt

d = dict()

f = open('hist_shmem.dat')

flag = 0
for line in f.readlines():
    if not flag:
        flag = 1
        continue
    z = line.split(' ')
    x = float(z[0])
    y = int(z[1])
    d[x] = y / 81920000
    
x1 = np.linspace(0, 24, num=1024, endpoint=False)
x2 = np.linspace(1.0/1024, 24, num=1024, endpoint=True)
xs = x1 + 1.0/2048
ys = np.exp(-x1) - np.exp(-x2)
ys = ys / np.sum(ys)

plt.plot(xs, ys, 'r')
plt.xlabel('X')
plt.ylabel('Frequency')
plt.bar(d.keys(), d.values(), width=24.0/1024)
plt.show()
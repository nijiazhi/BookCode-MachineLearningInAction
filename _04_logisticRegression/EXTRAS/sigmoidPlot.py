'''
Created on Oct 6, 2010

@author: Peter
'''

import sys
from pylab import *

# 画一下sigmoid函数
t = arange(-60.0, 60, 0.1)
s = 1/(1 + exp(-t))
ax = subplot(211)
ax.plot(t, s)
ax.axis([-5, 5, 0, 1])
plt.xlabel('x')
plt.ylabel('Sigmoid(x)')

ax = subplot(212)
ax.plot(t, s)
ax.axis([-60, 60, 0, 1])  # 横纵坐标
plt.xlabel('x')
plt.ylabel('Sigmoid(x)')

show()
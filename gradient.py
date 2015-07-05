__author__ = 'user'

import numpy as np
import matplotlib.pyplot as plt

def htheta(theta0,theta1,t):
    return [theta0 + theta1*x for x in t]


t = [i for i in range(100)]

plt.plot(t,htheta(0,10,t),'-')
plt.show()
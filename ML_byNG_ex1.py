__author__ = 'user'


import numpy as np
import matplotlib.pyplot as plt



def computeCost(X,y,theta):
    m = y.__len__()
    J = np.sum((np.dot(X,theta) - np.reshape(y,(m,1)))**2)/2/m
    return J


def gradientDescent(X, y, theta, alpha, iterations):
    Jhis = np.zeros([iteration,1])
    m = y.__len__()
    for i in range(iterations):
        theta0 = theta[0] - alpha * np.sum(np.dot(X,theta)-np.reshape(y,(m,1)))/m
        theta1 = theta[1] - alpha * np.sum((np.dot(X,theta) - np.reshape(y,(m,1)))*np.reshape(X[:,1],(m,1)))/m
        theta = np.array([theta0,theta1])
        Jhis[i] = computeCost(X,y,theta)
    return theta

A = np.eye(5)

data = np.loadtxt('ex1data1.txt',delimiter=',')
x = data[:,0]
y = data[:,1]
m = x.__len__()

X = np.hstack((np.ones([m,1]),np.reshape(data[:,0],(m,1))))
theta = np.zeros([2,1])

iteration = 1500
alpha = 0.001

c1 = computeCost(X, y, theta)
print(c1)

thetares = gradientDescent(X,y,theta,alpha,iteration)

it = [i for i in range(iteration)]
plt.plot(x,y,'rx')
plt.plot(x,np.dot(X,thetares),'-')
#plt.plot(it,Jhis,'rx')
plt.show()

predict1 = np.dot(np.array([1,3.5]),thetares)
predict2 = np.dot(np.array([1,7]),thetares)
print(predict1*10000)
print(predict2*10000)


theta0_val = np.linspace(-10,100,50)
theta1_val = np.linspace(-1,100,50)

Jv = [[],[]]

for i in range(theta0_val.__len__()):
    for j in range(theta1_val.__len__()):
        t = np.array([theta0_val[i],theta1_val[i]])
        Jv[i][j] = computeCost(X,y,t)




print('ok')





import matplotlib.pyplot as plt
import numpy as np
from Quadrotor import Quad
plt.style.use('seaborn-v0_8')


M = 0.468
g = 9.81
Ixx = 4.856 * 10 ** -3
Iyy = 4.856 * 10 ** -3
Izz = 8.81 * 10 ** -3
Ir = 3.357 * 10 ** -5
l = 0.225
k = 2.98 * 10 ** -6
b0 = 1.14 * 10 ** -7
b1 = 280.19
beta0 = 189.63
beta1 = 6.0612
beta2 = 0.0122
k1 = 0.5
p1 = 5 * 10 ** -7
Ax = 0.25
Ay = 0.25
Az = 0.25
trajectory = []
Q1 = Quad(M, Ixx, Iyy, Izz, Ir, l, k, b0, b1, beta0, beta1, beta2, k1, p1, Ax, Ay, Az, g)
# initial condition

State = [-5, -8, 3, 4, -10, 1, 5, 8, -1, 12, 0, 0.5, 600, 75.2, 250.2, 50.2]

teval = 0.25
timespan = [0, teval]
method = 'RK23'
Input = [600,100 ,200 ,250]
x_real = Q1.motion(State, Input, method, timespan, teval)


plt.figure(1)

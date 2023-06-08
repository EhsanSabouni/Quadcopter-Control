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

State = [0, 1, 0, 0, 0.01, 0, 0, 0, 0, 0, 0, 0, 622.2, 622.2, 622.2, 622.2]


t = np.linspace(0, 0.1)
Input = [625, 625, 625, 625]
x_real1 = Q1.motion1(State, Input, t)
#x_real2 = Q1.motion2(State, Input, t)

x_simulated = Q1.rk4(t, State, Input,2)

#plt.figure(1)
#plt.subplot(2, 1, 1)
#plt.plot(t, x[:, 0], label='x')
#plt.plot(t, x[:, 2], label='y')
#plt.plot(t, x[:, 4], label='z')
#plt.legend()
#plt.subplot(2, 1, 2)
#plt.plot(t, x[:, 1], label='vx')
#plt.plot(t, x[:, 3], label='vy')
#plt.plot(t, x[:, 5], label='vz')
#plt.legend()
#plt.xlabel('time')

#plt.figure(2)
#plt.axes(projection='3d').plot3D(x[:, 0], x[:, 2], x[:, 4])
#plt.xlabel('x')
#plt.xlabel('y')
#plt.show()
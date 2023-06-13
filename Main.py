import matplotlib.pyplot as plt
import numpy as np
from Quadrotor import Quad
import matplotlib.pyplot as plt
import time
plt.style.use('seaborn-v0_8')
plt.ion()
fig = plt.figure()
ax = plt.axes(projection='3d')

method = 'DOP853'
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
Q1 = Quad(M, Ixx, Iyy, Izz, Ir, l, k, b0, b1, beta0, beta1, beta2, k1, p1, Ax, Ay, Az, g, method)

#initial condition
positions = [0, 0, 0.01]  #x, y, z
angles = np.zeros(3)  #phi, theta, psi
omegas = [6.35/np.sqrt(k), 6.35/np.sqrt(k), 6.35/np.sqrt(k), (6.35-0.5)/np.sqrt(k)] #omega1, omega2, omega3, omega4
State = [positions[0], 0, positions[1], 0, positions[2], 0, angles[0], 0, angles[1], 0, angles[2], 0, omegas[0], omegas[1], omegas[2], omegas[3]]
# inputs
Input = [0, 0, 0, 0]
#simulation time
endtime = 1
steps = 20
t = np.linspace(0, endtime, num=steps)


Data_position = np.zeros((steps, 3))
Data_angles = np.zeros((steps, 3))
Data_linearvelocity = np.zeros((steps, 3))
Data_angularvelocity = np.zeros((steps, 3))
Data_inputs = np.zeros((steps, 4))
Data_position[0] = positions
Data_angles[0] = angles


for i in range(len(t)-1):
    timespan = [t[i], t[i+1]]
    teval = t[i+1]
    State = Q1.motion(Q1.dynamics, State, Input, method, timespan, teval)
    #State = Q1.rk4(Q1.dynamics, timespan, State, Input, 100)
    positions = [State[0], State[2], State[4]]
    angles = [State[6], State[8], State[10]]
    Data_position[i+1] = positions
    Data_angles[i+1] = angles
    Q1.animate(positions, angles, l, fig, ax)

plt.ioff()
fig2, axs2 = plt.subplots(2)
Q1.plotdata(t, Data_position, Data_angles, fig2, axs2)
plt.show()



import matplotlib.pyplot as plt
import numpy as np
from Quadrotor import Quad
import matplotlib.pyplot as plt
import time

plt.style.use('seaborn-v0_8')
animation = 0  # 1 for animation, 0 for no animation
if animation == 1:
    plt.ion()
    fig = plt.figure()
    ax = plt.axes(projection='3d')

contionus = 0  # 1 for contionus, 0 for discrete
if contionus == 0:
    discretesteps = 10

method = 'DOP853'
M = 0.468
g = 9.81
Ixx, Iyy, Izz = 4.856 * 10 ** -3, 4.856 * 10 ** -3, 8.801 * 10 ** -3
b0, b1 = 1.14 * 10 ** -7, 280.19
Ax, Ay, Az = 0.25, 0.25, 0.25
beta0, beta1, beta2 = 189.63, 6.0612, 0.0122
Ir = 3.357 * 10 ** -5
l = 0.225
k = 2.98 * 10 ** -6
k1 = 0.5
p1 = 5 * 10 ** -7

Q1 = Quad(M, Ixx, Iyy, Izz, Ir, l, k, b0, b1, beta0, beta1, beta2, k1, p1, Ax, Ay, Az, g, method)

# Simulation parameters
timeout = 100
stepsize = 0.1
steps = int(float(timeout / stepsize))
t = np.linspace(0, timeout, num=steps)
simindex = 0

#initial condition
x, y, z = np.array([20, -20, 1])
x_dot, y_dot, z_dot = np.array([0, 0, 0])
phi, theta, psi = np.array([0, 0, 0])
phi_dot, theta_dot, psi_dot = np.array([0, 0, 0])
p, q, r = np.array([0, 0, 0])
omega1, omega2, omega3, omega4 = np.array([622, 622, 622, 622])
state = [x, x_dot, y, y_dot, z, z_dot, phi, phi_dot, theta, theta_dot, psi, psi_dot, p, q, r, omega1, omega2, omega3, omega4]
input = np.array([0, 0, 0, 0, 0, 0, 0, 0])


desiredgoal = [42, -36, 17]
dist2goal = [np.sqrt((state[0] - desiredgoal[0]) ** 2 + (state[2] - desiredgoal[1]) ** 2 +
                     (state[4] - desiredgoal[2]) ** 2)]
thr = 1 # threshold for goal
obstacle = [35, -30, 15, 7]  # sphere xc,yc,zc,r

Data_position = [np.array([state[0], state[2], state[4]])]
Data_angles = [np.array([state[6], state[8], state[10]])]
Data_linearvelocity = [np.array([state[1], state[3], state[5]])]
Data_angularvelocity = [np.array([state[7], state[9], state[11]])]
Data_inputs = [np.array([input[0], input[1], input[2], input[3]])]


inputlimit = 0
vxlimit, vylimit, vzlimit = [1, 1, 1]
philimit, thetalimit = [1, 1]
xdclf, ydclf, zdclf = [0, 0, 1]
rdclf, phidclf, thetadclf = [1, 1, 1]
safetyconstraint = 1
constraintstatus = [inputlimit, vxlimit, vylimit, vzlimit, philimit, thetalimit,
                    zdclf, xdclf, ydclf, rdclf, phidclf, thetadclf,
                    safetyconstraint]




while dist2goal[-1] > thr and simindex < steps - 2:
    simindex += 1
    timespan = [t[simindex], t[simindex+1]]
    teval = t[simindex+1]
    input= Q1.qp_solver(state, desiredgoal, obstacle, constraintstatus)
    if contionus == 1:
        state = Q1.motion(Q1.dynamics, state, input[0:4], method, timespan, teval)
    else:
        state = Q1.rk4(Q1.dynamics, timespan, state, input[0:4], discretesteps)

    positions = [state[0], state[2], state[4]]
    angles = [state[6], state[8], state[10]]
    Data_inputs = np.append(Data_inputs, [input[0:4]], axis=0)
    Data_position = np.append(Data_position, [positions], axis=0)
    Data_angles = np.append(Data_angles, [angles], axis=0)
    if animation == 1:
        Q1.animate(positions, angles, obstacle, l, fig, ax)

    dist2goal = np.append(dist2goal, [np.sqrt((state[0] - desiredgoal[0]) ** 2 + (state[2] - desiredgoal[1]) ** 2 + (state[4] - desiredgoal[2]) ** 2)])


simlength = np.linspace(0, teval, num=simindex+1)

if animation == 1:
    plt.ioff()

fig2, axs2 = plt.subplots(3)
Q1.plotdata(simlength, Data_position, Data_angles, Data_inputs, fig2, axs2)
fig3 = plt.figure()
ax1 = fig3.add_subplot(projection='3d')
ax1.plot(Data_position[1:, 0], Data_position[1:, 1], Data_position[1:, 2])
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = obstacle[3]*np.cos(u)*np.sin(v)+obstacle[0]
y = obstacle[3]*np.sin(u)*np.sin(v)+obstacle[1]
z = obstacle[3]*np.cos(v)+obstacle[2]
ax1.plot_wireframe(x, y, z, color="red")
fig4 = plt.figure()
plt.plot(simlength, dist2goal)
plt.xlabel('time(s)')
plt.ylabel('distance to goal(m)')
plt.title('distance to goal')
plt.show()


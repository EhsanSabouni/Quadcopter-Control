import numpy as np
from scipy.integrate import odeint,solve_ivp,ode
import time


class Quad:
    def __init__(self, M: float, Ixx: float, Iyy: float, Izz: float, Ir: float, l: float, k: float,
                 b0: float, b1: float, beta0: float, beta1: float,
                 beta2: float, k1: float, p1: float, Ax: float, Ay: float, Az: float, g: float, method: str):
        self.M = M
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.Ir = Ir
        self.l = l
        self.k = k
        self.b0 = b0
        self.b1 = b1
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.k1 = k1
        self.p1 = p1
        self.Ax = Ax
        self.Ay = Ay
        self.Az = Az
        self.g = g
        self.method = method

    def dynamics(self, t, x):
        dx = np.zeros(20)  # [(x,0), (vx,1), (y,2), (vy,3),  (z,4), (vz,5), (phi,6), (omega_phi,7), (theta,8),
        # (omega_theta,9), (psi,10), (omega_psi,11), (omega1,12), (omega2,13), (omega3,14), (omega4,15),
        #   (u1,16), (u2,17), (u3,18), (u4,19)]

        tau_phi = self.l * self.k * (-x[13] ** 2 + x[15] ** 2)
        tau_theta = self.l * self.k * (-x[12] ** 2 + x[14] ** 2)
        tau_psi = self.b0 * (-x[12] ** 2 - x[14] ** 2 + x[13] ** 2 + x[15] ** 2)
        omega_gamma = x[12] - x[13] + x[14] - x[15]
        F = self.k * (x[12] ** 2 + x[13] ** 2 + x[14] ** 2 + x[15] ** 2)

        dx[0] = x[1]
        dx[1] = F / self.M * (
                    np.cos(x[10]) * np.sin(x[8]) * np.cos(x[6]) + np.sin(x[10]) * np.sin(x[6])) - 1 / self.M * self.Ax * \
                x[1]
        dx[2] = x[3]
        dx[3] = F / self.M * (
                    np.sin(x[10]) * np.sin(x[8]) * np.cos(x[6]) - np.cos(x[10]) * np.sin(x[6])) - 1 / self.M * self.Ay * \
                x[3]
        dx[4] = x[5]
        dx[5] = -self.g + F / self.M * (np.cos(x[8]) * np.cos(x[6])) - 1 / self.M * self.Az * x[5]

        D1 = np.array([[0, x[7] * np.cos(x[6]) * np.tan(x[8]) + x[9] * np.sin(x[6]) / (np.cos(x[8])) ** 2,
                        -x[7] * np.sin(x[6]) * np.cos(x[8]) + x[9] * np.cos(x[6]) / (np.cos(x[8])) ** 2],
                       [0, -x[7] * np.sin(x[6]), -x[7] * np.cos(x[6])],
                       [0, x[7] * np.cos(x[6]) / np.cos(x[8]) + x[7] * np.sin(x[6]) * np.tan(x[8]) / np.cos(x[8]),
                        -x[7] * np.sin(x[6]) / np.cos(x[8]) + x[9] * np.cos(x[6]) * np.tan(x[8]) / np.cos(x[8])]])

        W_eta_inv = np.array(
            [[1, np.sin(x[6]) * np.tan(x[8]), np.cos(x[6]) * np.tan(x[8])], [0, np.cos(x[6]), -np.sin(x[8])],
             [0, np.sin(x[6]) / np.cos(x[8]), np.cos(x[6]) / np.cos(x[8])]])

        W_eta = np.array([[1, 0, -np.sin(x[8])], [0, np.cos(x[6]), np.cos(x[8]) * np.sin(x[6])],
                          [0, -np.sin(x[6]), np.cos(x[8]) * np.cos(x[6])]])

        p, q, r = np.dot(W_eta, [x[7], x[9], x[11]])
        D2 = np.array(
            [(self.Iyy - self.Izz) * q * r / self.Ixx - self.Ir * (q / self.Ixx) * omega_gamma + tau_phi / self.Ixx,
             (self.Izz - self.Ixx) * p * r / self.Iyy - self.Ir * (-p / self.Iyy) * omega_gamma + tau_theta / self.Iyy,
             (self.Ixx - self.Iyy) * p * q / self.Izz + tau_psi / self.Izz])
        dx[6] = x[7]
        dx[7], dx[9], dx[11] = np.dot(D1, [p, q, r]) + np.dot(W_eta_inv, D2)

        dx[8] = x[9]
        dx[10] = x[11]

        dx[12] = self.b1 * x[16] - self.beta0 - self.beta1 * x[12] - self.beta2 * x[12] ** 2
        dx[13] = self.b1 * x[17] - self.beta0 - self.beta1 * x[13] - self.beta2 * x[13] ** 2
        dx[14] = self.b1 * x[18] - self.beta0 - self.beta1 * x[14] - self.beta2 * x[14] ** 2
        dx[15] = self.b1 * x[19] - self.beta0 - self.beta1 * x[15] - self.beta2 * x[15] ** 2

        dx[16:20] = [0, 0, 0, 0]

        return dx

    def motion(self, dynamics, state, input, method, timespan, teval):
        y0 = np.append(state, input)
        sol = solve_ivp(dynamics, timespan, y0, method=method, t_eval=[teval])
        #x = sol.y
        x = np.reshape(sol.y[0:16], 16)
        return x

    def rk4(self, dynamics, t, state, input, n):
        state = np.append(state, input)
        # Calculating step size
        #x0 = np.append(state, input)
        h = (t[-1] - t[0]) / n
        t0 = t[0]
        for i in range(n):
            k1 = (dynamics(t0, state))
            k2 = (dynamics((t0 + h / 2), (state + h * k1 / 2)))
            k3 = (dynamics((t0 + h / 2), (state + h * k2 / 2)))
            k4 = (dynamics((t0 + h), (state + h * k3)))
            k = h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            xn = state + k
            state = xn
            t0 = t0 + h

        return xn
    def discretedynamics(self, dt: float, x, u):
        # [(x,0), (vx,1), (y,2), (vy,3),  (z,4), (vz,5), (phi,6), (omega_phi,7), (theta,8),
        # (omega_theta,9), (psi,10), (omega_psi,11), (omega1,12), (omega2,13), (omega3,14), (omega4,15),
        #   (u1,16), (u2,17), (u3,18), (u4,19)]

        tau_phi = self.l * self.k * (-x[13] ** 2 + x[15] ** 2)
        tau_theta = self.l * self.k * (-x[12] ** 2 + x[14] ** 2)
        tau_psi = self.b0 * (-x[12] ** 2 - x[14] ** 2 + x[13] ** 2 + x[15] ** 2)
        omega_gamma = x[12] - x[13] + x[14] - x[15]
        F = self.k * (x[12] ** 2 + x[13] ** 2 + x[14] ** 2 + x[15] ** 2)

        x[0] += dt * x[1]
        x[1] += dt * (F / self.M * (
                    np.cos(x[10]) * np.sin(x[8]) * np.cos(x[6]) + np.sin(x[10]) * np.sin(x[6])) - 1 / self.M * self.Ax * x[1])
        x[2] += dt * x[3]
        x[3] += dt * (F / self.M * ( np.sin(x[10]) * np.sin(x[8]) * np.cos(x[6]) - np.cos(x[10]) * np.sin(x[6])) - 1 / self.M * self.Ay * x[3])
        x[4] += dt * x[5]
        x[5] += dt * (-self.g + F / self.M * (np.cos(x[8]) * np.cos(x[6])) - 1 / self.M * self.Az * x[5])

        D1 = np.array([[0, x[7] * np.cos(x[6]) * np.tan(x[8]) + x[9] * np.sin(x[6]) / (np.cos(x[8])) ** 2,
                        -x[7] * np.sin(x[6]) * np.cos(x[8]) + x[9] * np.cos(x[6]) / (np.cos(x[8])) ** 2],
                       [0, -x[7] * np.sin(x[6]), -x[7] * np.cos(x[6])],
                       [0, x[7] * np.cos(x[6]) / np.cos(x[8]) + x[7] * np.sin(x[6]) * np.tan(x[8]) / np.cos(x[8]),
                        -x[7] * np.sin(x[6]) / np.cos(x[8]) + x[9] * np.cos(x[6]) * np.tan(x[8]) / np.cos(x[8])]])

        W_eta_inv = np.array(
            [[1, np.sin(x[6]) * np.tan(x[8]), np.cos(x[6]) * np.tan(x[8])], [0, np.cos(x[6]), -np.sin(x[8])],
             [0, np.sin(x[6]) / np.cos(x[8]), np.cos(x[6]) / np.cos(x[8])]])

        W_eta = np.array([[1, 0, -np.sin(x[8])], [0, np.cos(x[6]), np.cos(x[8]) * np.sin(x[6])],
                          [0, -np.sin(x[6]), np.cos(x[8]) * np.cos(x[6])]])

        p, q, r = np.dot(W_eta, [x[7], x[9], x[11]])
        D2 = np.array(
            [(self.Iyy - self.Izz) * q * r / self.Ixx - self.Ir * (q / self.Ixx) * omega_gamma + tau_phi / self.Ixx,
             (self.Izz - self.Ixx) * p * r / self.Iyy - self.Ir * (-p / self.Iyy) * omega_gamma + tau_theta / self.Iyy,
             (self.Ixx - self.Iyy) * p * q / self.Izz + tau_psi / self.Izz])
        x[6] += dt * x[7]
        x[7], x[9], x[11] = [x[7], x[9], x[11]] + dt * (np.dot(D1, [p, q, r]) + np.dot(W_eta_inv, D2))

        x[8] += x[9]
        x[10] += x[11]

        x[12] += dt * (self.b1 * u[0] - self.beta0 - self.beta1 * x[12] - self.beta2 * x[12] ** 2)
        x[13] += dt * (self.b1 * u[1] - self.beta0 - self.beta1 * x[13] - self.beta2 * x[13] ** 2)
        x[14] += dt * (self.b1 * u[2] - self.beta0 - self.beta1 * x[14] - self.beta2 * x[14] ** 2)
        x[15] += dt * (self.b1 * u[3] - self.beta0 - self.beta1 * x[15] - self.beta2 * x[15] ** 2)

        return x

    def get_rotation(self, phi, theta, psi):

        R_x = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
        R_y = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        R_z = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
        R_zy = np.dot(R_z, R_y)
        R_zyx = np.dot(R_zy, R_x)

        return R_zyx

    def body2world(self, R, axle, position, flag):
        row_axle, col_axle = np.shape(axle)
        new_axle = np.zeros((row_axle, col_axle))
        for i in range(row_axle):
            r_body = axle[i]
            r_world = np.dot(R, r_body)
            new_axle[i] = r_world
            if flag == 1:
                new_axle[i] += position

        return new_axle

    def animate(self, position, angles, l, fig, ax):
        x = position[0]
        y = position[1]
        z = position[2]
        phi = angles[0]
        theta = angles[1]
        psi = angles[2]
        r = 0.1 * l
        ang = np.linspace(0, 2 * np.pi)
        axle_x = np.array([[-0.5, 0, 0], [0.5, 0, 0]])
        axle_y = np.array([[0, -0.5, 0], [0, 0.5, 0]])
        x_circle = r * np.cos(ang)
        y_circle = r * np.sin(ang)
        z_circle = np.zeros(len(ang))
        propeller = np.transpose([x_circle, y_circle, z_circle])
        p1, q1 = np.shape(propeller)
        p2, q2 = np.shape(axle_x)
        R = self.get_rotation(phi, theta, psi)

        new_axle_x = self.body2world(R, axle_x, position, flag=1)
        new_axle_y = self.body2world(R, axle_y, position, flag=1)
        new_propeller = self.body2world(R, propeller, position, flag=0)

        new_propeller1 = new_propeller + new_axle_x[0]
        new_propeller2 = new_propeller + new_axle_x[1]
        new_propeller3 = new_propeller + new_axle_y[0]
        new_propeller4 = new_propeller + new_axle_y[1]
        ax.axes.set_xlim3d(left=-2, right=2)
        ax.axes.set_ylim3d(bottom=-2, top=2)
        ax.axes.set_zlim3d(bottom=-2, top=2)
        ax.plot(new_axle_x[:, 0], new_axle_x[:, 1], new_axle_x[:, 2], 'b')
        ax.plot(new_axle_y[:, 0], new_axle_y[:, 1], new_axle_y[:, 2], 'b')
        ax.plot(new_propeller1[:, 0], new_propeller1[:, 1], new_propeller1[:, 2], 'r')
        ax.plot(new_propeller2[:, 0], new_propeller2[:, 1], new_propeller2[:, 2], 'r')
        ax.plot(new_propeller3[:, 0], new_propeller3[:, 1], new_propeller3[:, 2], 'r')
        ax.plot(new_propeller4[:, 0], new_propeller4[:, 1], new_propeller4[:, 2], 'r')
        fig.canvas.draw()
        time.sleep(0.1)
        fig.canvas.flush_events()
        ax.clear()
        return None

    def plotdata(self, t, Data_position, Data_angles, fig2, axs2):

        axs2[0].plot(t, np.array(Data_position)[:, 0], label='x')
        axs2[0].plot(t, np.array(Data_position)[:, 1], label='y')
        axs2[0].plot(t, np.array(Data_position)[:, 2], label='z')
        axs2[0].legend()
        axs2[0].set_xlabel('time [s]')
        axs2[0].set_ylabel('position [m]')
        axs2[1].plot(t, np.array(Data_angles)[:, 0], label='phi')
        axs2[1].plot(t, np.array(Data_angles)[:, 1], label='theta')
        axs2[1].plot(t, np.array(Data_angles)[:, 2], label='psi')
        axs2[1].legend()
        axs2[1].set_xlabel('time [s]')
        axs2[1].set_ylabel('angle [rad]')


        return None






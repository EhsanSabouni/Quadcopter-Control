import numpy as np
from scipy.integrate import odeint,solve_ivp,ode
import time
from cvxopt import matrix
from cvxopt.solvers import qp
import casadi as cd
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
        self.armlength = 2.5
        self.method = method

    def dynamics(self, t, x):
        dx = [0] * 23  # [(x,0), (vx,1), (y,2), (vy,3),  (z,4), (vz,5), (phi,6), (omega_phi,7), (theta,8),
        # (omega_theta,9), (psi,10), (omega_psi,11), (p, 12), (q,13), (r, 14) (omega1,15), (omega2,16), (omega3,17), (omega4,18),
        #   (u1,19), (u2,20), (u3,21), (u4,2)]

        tau_phi = self.l * self.k * (x[18] ** 2 - x[16] ** 2)
        tau_theta = self.l * self.k * (x[17] ** 2 - x[15] ** 2)
        tau_psi = self.b0 * (x[16] ** 2 + x[18] ** 2 - x[15] ** 2 - x[17] ** 2)
        omega_gamma = x[15] - x[16] + x[17] - x[18]
        F = self.k * (x[15] ** 2 + x[16] ** 2 + x[17] ** 2 + x[18] ** 2)


        dx[0] = x[1]
        dx[1] = F / self.M * (
                np.cos(x[10]) * np.sin(x[8]) * np.cos(x[6]) + np.sin(x[10]) * np.sin(x[6])) - self.Ax * x[1]
        dx[2] = x[3]
        dx[3] = F / self.M * (
                np.sin(x[10]) * np.sin(x[8]) * np.cos(x[6]) - np.cos(x[10]) * np.sin(x[6])) - self.Ay * x[3]
        dx[4] = x[5]
        dx[5] = -self.g + F / self.M * (np.cos(x[8]) * np.cos(x[6])) -  self.Az * x[5]

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
        dx[12] = p
        dx[13] = q
        dx[14] = r
        dx[15] = self.b1 * x[19] - self.beta0 - self.beta1 * x[15] - self.beta2 * x[15] ** 2
        dx[16] = self.b1 * x[20] - self.beta0 - self.beta1 * x[16] - self.beta2 * x[16] ** 2
        dx[17] = self.b1 * x[21] - self.beta0 - self.beta1 * x[17] - self.beta2 * x[17] ** 2
        dx[18] = self.b1 * x[22] - self.beta0 - self.beta1 * x[18] - self.beta2 * x[18] ** 2

        dx[19:23] = [0, 0, 0, 0]
        return dx

    def dynamics_f(self, x):

        tau_phi = self.l * self.k * (-x[13] ** 2 + x[15] ** 2)
        tau_theta = self.l * self.k * (-x[12] ** 2 + x[14] ** 2)
        tau_psi = self.b0 * (-x[12] ** 2 - x[14] ** 2 + x[13] ** 2 + x[15] ** 2)
        omega_gamma = x[12] - x[13] + x[14] - x[15]
        F = self.k * (x[12] ** 2 + x[13] ** 2 + x[14] ** 2 + x[15] ** 2)

        f_0 = x[1]
        f_1 = F / self.M * (
                    np.cos(x[10]) * np.sin(x[8]) * np.cos(x[6]) + np.sin(x[10]) * np.sin(x[6])) - 1 / self.M * self.Ax * \
                x[1]
        f_2 = x[3]
        f_3 = F / self.M * (
                    np.sin(x[10]) * np.sin(x[8]) * np.cos(x[6]) - np.cos(x[10]) * np.sin(x[6])) - 1 / self.M * self.Ay * \
                x[3]
        f_4 = x[5]
        f_5 = -self.g + F / self.M * (np.cos(x[8]) * np.cos(x[6])) - 1 / self.M * self.Az * x[5]

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
        f_6 = x[7]
        f_7, f_9, f_11 = np.dot(D1, [p, q, r]) + np.dot(W_eta_inv, D2)

        f_8 = x[9]
        f_10 = x[11]

        f_12 = - self.beta0 - self.beta1 * x[12] - self.beta2 * x[12] ** 2
        f_13 = - self.beta0 - self.beta1 * x[13] - self.beta2 * x[13] ** 2
        f_14 = - self.beta0 - self.beta1 * x[14] - self.beta2 * x[14] ** 2
        f_15 = - self.beta0 - self.beta1 * x[15] - self.beta2 * x[15] ** 2


        f = [f_0, f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8, f_9, f_10, f_11, f_12, f_13, f_14, f_15]

        return f

    def dynamics_g(self, x):
        g = np.zeros((16, 4))
        g[15, 3], g[14, 2], g[13, 1], g[12, 0] = [self.b1, self.b1, self.b1, self.b1]
        return g

    def motion(self, dynamics, state, input, method, timespan, teval):
        y0 = np.append(state, input)
        AU = 1.5e11
        a = AU
        sol = solve_ivp(dynamics, timespan, y0, method=method, t_eval=[teval], atol=1e-6*a)
        #x = sol.y
        x = np.reshape(sol.y[0:len(state)], len(state))
        return x

    def rk4(self, dynamics, t, state, input, n):
        statesnumber = len(state)
        state = np.append(state, input)
        # Calculating step size
        #x0 = np.append(state, input)
        h = np.array([(t[-1] - t[0]) / n])
        t0 = t[0]
        for i in range(n):
            k1 = np.array(dynamics(t0, state))
            k2 = np.array(dynamics((t0 + h / 2), (state + h * k1 / 2)))
            k3 = np.array(dynamics((t0 + h / 2), (state + h * k2 / 2)))
            k4 = np.array(dynamics((t0 + h), (state + h * k3)))
            k = np.array(h * (k1 + 2 * k2 + 2 * k3 + k4) / 6)
            xn = state + k
            state = xn
            t0 = t0 + h

        return xn[0:statesnumber]

    def rk4_casadi(self, dynamics, t, state, input, n):
        state = np.append(state, input)
        # Calculating step size
        h = float((t[-1] - t[0]) / n)
        t0 = t[0]

        for i in range(n):
            k = []
            k1 = []
            xn = []
            for j in range(len(state)):
                k1.append((dynamics(t0, state))[j])
                # k2.append((dynamics((t0 + h / 2), (state + h * k1 / 2)))[j])
                # k3.append((dynamics((t0 + h / 2), (state + h * k2 / 2)))[j])
                # k4.append((dynamics((t0 + h), (state + h * k3)))[j])
                # k.append(h * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6)
                k.append(h * k1[j])
                xn.append(state[j] + k[j])
            state = xn
            t0 = t0 + h

        return xn[0:16]

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

    def animate(self, position, angles, obstacle, l, fig, ax):
        x = position[0]
        y = position[1]
        z = position[2]
        phi = angles[0]
        theta = angles[1]
        psi = angles[2]
        r = 0.1 * l
        ang = np.linspace(0, 2 * np.pi)
        axle_x = np.array([[-self.armlength, 0, 0], [self.armlength, 0, 0]])
        axle_y = np.array([[0, -self.armlength, 0], [0, self.armlength, 0]])
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
        ax.axes.set_xlim3d(left=-40, right=40)
        ax.axes.set_ylim3d(bottom=-40, top=40)
        ax.axes.set_zlim3d(bottom=0, top=20)
        ax.plot(new_axle_x[:, 0], new_axle_x[:, 1], new_axle_x[:, 2], 'b')
        ax.plot(new_axle_y[:, 0], new_axle_y[:, 1], new_axle_y[:, 2], 'b')
        ax.plot(new_propeller1[:, 0], new_propeller1[:, 1], new_propeller1[:, 2], 'r')
        ax.plot(new_propeller2[:, 0], new_propeller2[:, 1], new_propeller2[:, 2], 'r')
        ax.plot(new_propeller3[:, 0], new_propeller3[:, 1], new_propeller3[:, 2], 'r')
        ax.plot(new_propeller4[:, 0], new_propeller4[:, 1], new_propeller4[:, 2], 'r')
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = obstacle[3] * np.cos(u) * np.sin(v) + obstacle[0]
        y = obstacle[3] * np.sin(u) * np.sin(v) + obstacle[1]
        z = obstacle[3] * np.cos(v) + obstacle[2]
        ax.plot_wireframe(x, y, z, color="red")
        fig.canvas.draw()
        time.sleep(0.1)
        fig.canvas.flush_events()
        ax.clear()
        return None

    def plotdata(self, t, Data_position, Data_angles, Data_inputs, fig2, axs2):

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
        axs2[2].plot(t[1:], np.array(Data_inputs)[1:, 0], label='u1')
        axs2[2].plot(t[1:], np.array(Data_inputs)[1:, 1], label='u2')
        axs2[2].plot(t[1:], np.array(Data_inputs)[1:, 2], label='u3')
        axs2[2].plot(t[1:], np.array(Data_inputs)[1:, 3], label='u4')
        axs2[2].legend()
        axs2[2].set_xlabel('time [s]')
        axs2[2].set_ylabel('inputs [rad]')


        return None


    def cbf_calculator (self, b, x, alphas, cbf_order, num_inputs, num_states):
        # phid = 6
        # R0 = 10
        # xd = 1
        # yd = 2
        # zd = 3
        # x = cd.MX.sym('x', 16)
        # b = (x[0] - xd) ** 2 + (x[2] - yd) ** 2 + (x[4] - zd) ** 2 - R0 ** 2
        lgb = cd.MX.zeros(num_inputs, 1)
        for m in range(0, cbf_order):
            lfb = 0
            lgb = 0
            db_dx = cd.jacobian(b, x)
            for i in range(0,num_states):
                lfb = lfb + db_dx[i] * self.dynamics_f(x)[i]
                lgb = lgb + db_dx[i] @ self.dynamics_g(x)[i, :]
            b = lfb + alphas[m - 1] * b
        lgb = lgb.T
        lfb = cd.Function('cbf', [x], [b])
        lgb = cd.Function('lgb', [x], [lgb])
        return lfb, lgb

    def qp_solver(self, state, desiredgoal, obstacle, constraintstatus):
        x, y, z = state[0], state[2], state[4]
        x_dot, y_dot, z_dot = state[1], state[3], state[5]
        phi, theta, psi = state[6], state[8], state[10]
        phi_dot, theta_dot, psi_dot = state[7], state[9], state[11]
        ip, iq, ir = state[12], state[13], state[14]
        w1, w2, w3, w4 = state[15], state[16], state[17], state[18]

        x_d, y_d, z_d = desiredgoal[0], desiredgoal[1], desiredgoal[2]

        decay = 35
        phi_max = np.pi / 2
        phi_min = -np.pi / 2
        theta_max = np.pi / 2
        theta_min = -np.pi / 2
        #
        dis = np.sqrt((y_d - y) ** 2 + (x_d - x) ** 2)
        if psi < -np.pi:
            psi = np.pi
            state[10] = np.pi
        if (psi > np.pi):
            psi = -np.pi
            state[10] = -np.pi

        if ir < -np.pi:
            ir = np.pi
            state[14] = np.pi
        if (ir > np.pi):
            ir = -np.pi
            state[14] = -np.pi
        if dis >= decay:
            amplitude = np.pi / 24
        else:
            amplitude = np.pi / 24 * (dis / decay)

        heading_dr = np.arctan2((y_d - y), (x_d - x))
        heading_d = heading_dr - ir
        if heading_d < 0:
            heading_d = heading_d + 2 * np.pi
        h_pi = np.pi / 2
        if (heading_d >= 0 and heading_d <= h_pi):
            theta_d = amplitude * (h_pi - heading_d) / h_pi
            phi_d = -amplitude * heading_d / h_pi
        else:
            if (heading_d > h_pi and heading_d <= np.pi):
                theta_d = -amplitude * (heading_d - h_pi) / h_pi
                phi_d = -amplitude * (np.pi - heading_d) / h_pi
            else:
                if (heading_d > np.pi and heading_d < 3 * h_pi):
                    theta_d = -amplitude * (3 * h_pi - heading_d) / h_pi
                    phi_d = amplitude * (heading_d - np.pi) / h_pi
                else:
                    theta_d = amplitude * (heading_d - 3 * h_pi) / h_pi
                    phi_d = amplitude * (2 * np.pi - heading_d) / h_pi
        if (dis < 10):
            decay = 20

        Fr1 = self.beta0 + self.beta1 * w1 + self.beta2 * w1 ** 2
        Fr2 = self.beta0 + self.beta1 * w2 + self.beta2 * w2 ** 2
        Fr3 = self.beta0 + self.beta1 * w3 + self.beta2 * w3 ** 2
        Fr4 = self.beta0 + self.beta1 * w4 + self.beta2 * w4 ** 2
        #
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_psi = np.sin(psi)
        cos_psi = np.cos(psi)
        tao_phi = self.l * self.k * (w4 ** 2 - w2 ** 2)
        tao_theta = self.l * self.k * (w3 ** 2 - w1 ** 2)
        tao_psi = self.b0 * (w2 ** 2 + w4 ** 2 - w1 ** 2 - w3 ** 2)
        w_t = w1 - w2 + w3 - w4

        p = phi_dot - sin_theta * psi_dot
        q = cos_phi * theta_dot + cos_theta * sin_phi * psi_dot
        r = -sin_phi * theta_dot + cos_theta * cos_phi * psi_dot
        p_dot = tao_phi / self.Ixx + (self.Iyy - self.Izz) * q * r / self.Ixx - self.Ir * w_t * q / self.Ixx
        q_dot = tao_theta / self.Iyy + (self.Izz - self.Ixx) * p * r / self.Iyy + self.Ir * w_t * p / self.Iyy
        r_dot = tao_psi / self.Izz + (self.Ixx - self.Iyy) * p * q / self.Izz
        T = self.k * (w1 ** 2 + w2 ** 2 + w3 ** 2 + w4 ** 2)
        x_ddotcoe = cos_psi * sin_theta * cos_phi + sin_psi * sin_phi
        y_ddotcoe = sin_psi * sin_theta * cos_phi - cos_psi * sin_phi
        z_ddotcoe = cos_theta * cos_phi
        x_ddot = x_ddotcoe * T / self.M - self.Ax * x_dot
        y_ddot = y_ddotcoe * T / self.M - self.Ay * y_dot
        z_ddot = z_ddotcoe * T / self.M - self.Az * z_dot - self.g
        rotor_remain = 2 * w1 * (-Fr1) + 2 * w2 * (-Fr2) + 2 * w3 * (-Fr3) + 2 * w4 * (-Fr4)
        w_t_remain = (-Fr1) - (-Fr2) + (-Fr3) - (-Fr4)
        tao_phi_remain = 2 * w4 * self.l * self.k * (-Fr4) - 2 * w2 * self.l * self.k * (-Fr2)
        tao_theta_remain = 2 * w3 * self.l * self.k * (-Fr3) - 2 * w1 * self.l * self.k * (-Fr1)
        tao_psi_remain = 2 * w2 * self.b0 * (-Fr2) + 2 * w4 * self.b0 * (-Fr4) - 2 * w1 * self.b0 * (-Fr1) - 2 * w3 * self.b0 * (
            -Fr3)

        if constraintstatus[0] == 1:
            G = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0, 0, 0]])
            h = np.array([100, 100])

            G = np.append(G, [[0, 1, 0, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0, 0, 0]], axis=0)
            h = np.append(h, [100, 100])

            G = np.append(G, [[0, 0, 1, 0, 0, 0, 0, 0], [0, 0, -1, 0, 0, 0, 0, 0]], axis=0)
            h = np.append(h, [100, 100])

            G = np.append(G, [[0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, -1, 0, 0, 0, 0]], axis=0)
            h = np.append(h, [100, 100])
        else:
            G = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
            h = np.array([0, 0])

        #     'CBF for vz'

        v_zmax = 8
        pe = 1
        b = v_zmax - z_dot
        b_dot = -z_ddot
        Lf2bz = -(
                    -theta_dot * sin_theta * cos_phi - phi_dot * cos_theta * sin_phi) * T / self.M + self.Az * z_ddot - self.k * z_ddotcoe * rotor_remain / self.M
        #
        #
        b_vzmax = Lf2bz + 2 * pe * b_dot + pe ** 2 * b
        A1 = self.k * z_ddotcoe * 2 * w1 * self.b1 / self.M
        A2 = self.k * z_ddotcoe * 2 * w2 * self.b1 / self.M
        A3 = self.k * z_ddotcoe * 2 * w3 * self.b1 / self.M
        A4 = self.k * z_ddotcoe * 2 * w4 * self.b1 / self.M
        A_vzmax = [A1, A2, A3, A4, 0, 0, 0, 0]
        if constraintstatus[3] == 1:
            # G = np.array([A_vzmax])
            # h = np.array([b_vzmax])
            G = np.append(G, [A_vzmax], axis=0)
            h = np.append(h, [b_vzmax])
        #
        #
        v_zmin = -20
        pe = 1
        b = z_dot - v_zmin
        b_dot = z_ddot
        Lf2b = -Lf2bz
        b_vzmin = Lf2b + 2 * pe * b_dot + pe ** 2 * b
        #     print(b,z_dot, b_dot,Lf2b, b_vzmin)
        A_vzmin = [-A1, -A2, -A3, -A4, 0, 0, 0, 0]
        if constraintstatus[3] == 1:
            G = np.append(G, [A_vzmin], axis=0)
            h = np.append(h, [b_vzmin])

        #     'CBF for vx'
        v_xmax = 2
        pe = 1
        b = v_xmax - x_dot
        b_dot = -x_ddot
        Lf2bx = -(
                -psi_dot * sin_psi * sin_theta * cos_phi + theta_dot * cos_psi * cos_theta * cos_phi - phi_dot * cos_psi * sin_theta * sin_phi + psi_dot * cos_psi * sin_phi + phi_dot * sin_psi * cos_phi) * T / self.M + self.Ax * x_ddot - self.k * x_ddotcoe * rotor_remain / self.M
        A1 = self.k * x_ddotcoe * 2 * w1 * self.b1 / self.M
        A2 = self.k * x_ddotcoe * 2 * w2 * self.b1 / self.M
        A3 = self.k * x_ddotcoe * 2 * w3 * self.b1 / self.M
        A4 = self.k * x_ddotcoe * 2 * w4 * self.b1 / self.M
        A_vxmax = [A1, A2, A3, A4, 0, 0, 0, 0]
        b_vxmax = Lf2bx + 2 * pe * b_dot + pe ** 2 * b
        if constraintstatus[1] == 1:
            G = np.append(G, [A_vxmax], axis=0)
            h = np.append(h, [b_vxmax])
            # #
        v_xmin = -2
        pe = 1
        b = x_dot - v_xmin
        b_dot = x_ddot
        Lf2b = -Lf2bx
        b_vxmin = Lf2b + 2 * pe * b_dot + pe ** 2 * b
        A_vxmin = [-A1, -A2, -A3, -A4, 0, 0, 0, 0]
        if constraintstatus[1] == 1:
            G = np.append(G, [A_vxmin], axis=0)
            h = np.append(h, [b_vxmin])

        'CBF for vy'
        v_ymax = 2
        pe = 1
        b = v_ymax - y_dot
        b_dot = -y_ddot
        Lf2by = -(
                psi_dot * cos_psi * sin_theta * cos_phi + theta_dot * sin_psi * cos_theta * cos_phi - phi_dot * sin_psi * sin_theta * sin_phi + psi_dot * sin_psi * sin_phi - phi_dot * cos_psi * cos_phi) * T / self.M + self.Ay * y_ddot - self.k * y_ddotcoe * rotor_remain / self.M
        b_vymax = Lf2by + 2 * pe * b_dot + pe ** 2 * b
        A1 = self.k * y_ddotcoe * 2 * w1 * self.b1 / self.M
        A2 = self.k * y_ddotcoe * 2 * w2 * self.b1 / self.M
        A3 = self.k * y_ddotcoe * 2 * w3 * self.b1 / self.M
        A4 = self.k * y_ddotcoe * 2 * w4 * self.b1 / self.M
        A_vymax = [A1, A2, A3, A4, 0, 0, 0, 0]
        print(Lf2by)
        if constraintstatus[2] == 1:
            G = np.append(G, [A_vymax], axis=0)
            h = np.append(h, [b_vymax])
        # #
        v_ymin = -2
        pe = 1
        b = y_dot - v_ymin
        b_dot = y_ddot
        Lf2b = -Lf2by
        b_vymin = Lf2b + 2 * pe * b_dot + pe ** 2 * b
        A_vymin = [-A1, -A2, -A3, -A4, 0, 0, 0, 0]
        if constraintstatus[2] == 1:
            G = np.append(G, [A_vymin], axis=0)
            h = np.append(h, [b_vymin])

        'CBF for phi'
        pe = 1
        b = phi_max - ip
        b_dot = -p
        b_ddot = -p_dot
        Lf3b = -((self.Iyy - self.Izz) * (
                q_dot * r + q * r_dot) / self.Ixx - self.Ir * q_dot * w_t / self.Ixx - self.Ir * q * w_t_remain / self.Ixx + tao_phi_remain / self.Ixx)
        b_rollmax = Lf3b + 3 * pe * b_ddot + 3 * pe ** 2 * b_dot + pe ** 3 * b
        A1 = -self.Ir * q * self.b1 / self.Ixx
        A2 = self.Ir * q * self.b1 / self.Ixx - 2 * w2 * self.l * self.k * self.b1 / self.Ixx
        A3 = A1
        A4 = self.Ir * q * self.b1 / self.Ixx + 2 * w4 * self.l * self.k * self.b1 / self.Ixx
        A_rollmax = [A1, A2, A3, A4, 0, 0, 0, 0]
        print(b, b_dot, b_ddot, Lf3b, b_rollmax, A1, A2, A3, A4)
        if constraintstatus[4] == 1:
            G = np.append(G, [A_rollmax], axis=0)
            h = np.append(h, [b_rollmax])

        pe = 1
        b = ip - phi_min
        b_dot = p
        b_ddot = p_dot
        Lf3b = -Lf3b
        b_rollmin = Lf3b + 3 * pe * b_ddot + 3 * pe ** 2 * b_dot + pe ** 3 * b
        A_rollmin = [-A1, -A2, -A3, -A4, 0, 0, 0, 0]
        if constraintstatus[4] == 1:
            G = np.append(G, [A_rollmin], axis=0)
            h = np.append(h, [b_rollmin])
        #
        'CBF for theta'
        #
        pe = 1
        b = theta_max - iq
        b_dot = -q
        b_ddot = -q_dot
        Lf3b = -((self.Izz - self.Ixx) * (
                p_dot * r + p * r_dot) / self.Iyy + self.Ir * p_dot * w_t / self.Iyy + self.Ir * p * w_t_remain / self.Iyy + tao_theta_remain / self.Iyy)
        b_pitchmax = Lf3b + 3 * pe * b_ddot + 3 * pe ** 2 * b_dot + pe ** 3 * b
        A1 = self.Ir * p * self.b1 / self.Iyy - 2 * w1 * self.l * self.k * self.b1 / self.Iyy
        A2 = -self.Ir * p * self.b1 / self.Iyy
        A3 = self.Ir * p * self.b1 / self.Iyy + 2 * w3 * self.l * self.k * self.b1 / self.Iyy
        A4 = A2
        A_pitchmax = [A1, A2, A3, A4, 0, 0, 0, 0]
        if constraintstatus[5] == 1:
            G = np.append(G, [A_pitchmax], axis=0)
            h = np.append(h, [b_pitchmax])

        pe = 1
        b = iq - theta_min
        b_dot = q
        b_ddot = q_dot
        Lf3b = -Lf3b
        b_pitchmin = Lf3b + 3 * pe * b_ddot + 3 * pe ** 2 * b_dot + pe ** 3 * b
        A_pitchmin = [-A1, -A2, -A3, -A4, 0, 0, 0, 0]
        if constraintstatus[5] == 1:
            G = np.append(G, [A_pitchmin], axis=0)
            h = np.append(h, [b_pitchmin])
        #
        'Clf for zd'
        eps = 10
        k_feed1 = 1
        k_feed2 = 1
        var = z_ddot + k_feed1 * (z - z_d) + k_feed2 * z_dot
        V = var ** 2
        LfV = 2 * var * (k_feed1 * z_dot + k_feed2 * z_ddot + (
                -theta_dot * sin_theta * cos_phi - phi_dot * cos_theta * sin_phi) * T / self.M - self.Az * z_ddot + self.k * z_ddotcoe * rotor_remain / self.M)
        b_stabz = -LfV - eps * V
        if constraintstatus[6] == 1:
            G = np.append(G, [
                [2 * var * A_vzmax[0], 2 * var * A_vzmax[1], 2 * var * A_vzmax[2], 2 * var * A_vzmax[3], -1, 0, 0, 0]],
                          axis=0)
            h = np.append(h, [b_stabz])

        'Clf for xd'

        k_feed1 = 1
        k_feed2 = 1
        eps = 10
        var = x_ddot + k_feed1 * (x - x_d) + k_feed2 * x_dot
        V = var ** 2
        LfV = 2 * var * (k_feed1 * x_dot + k_feed2 * x_ddot + (
                -psi_dot * sin_psi * sin_theta * cos_phi + theta_dot * cos_psi * cos_theta * cos_phi - phi_dot * cos_psi * sin_theta * sin_phi + psi_dot * cos_psi * sin_phi + phi_dot * sin_psi * cos_phi) * T / self.M - self.Ax * x_ddot + self.k * x_ddotcoe * rotor_remain / self.M)
        b_stabx = -LfV - eps * V
        if constraintstatus[7] == 1:
            G = np.append(G, [
                [2 * var * A_vxmax[0], 2 * var * A_vxmax[1], 2 * var * A_vxmax[2], 2 * var * A_vxmax[3], 0, 0, -1, 0]],
                          axis=0)
            h = np.append(h, [b_stabx])
        #
        'Clf for yd'
        eps = 10
        k_feed1 = 1
        k_feed2 = 1
        var = y_ddot + k_feed1 * (y - y_d) + k_feed2 * y_dot
        V = var ** 2
        LfV = 2 * var * (k_feed1 * z_dot + k_feed2 * z_ddot + (
                psi_dot * cos_psi * sin_theta * cos_phi + theta_dot * sin_psi * cos_theta * cos_phi - phi_dot * sin_psi * sin_theta * sin_phi + psi_dot * sin_psi * sin_phi - phi_dot * cos_psi * cos_phi) * T / self.M - self.Ay * y_ddot + self.k * y_ddotcoe * rotor_remain / self.M)
        b_staby = -LfV - eps * V
        if constraintstatus[8] == 1:
            G = np.append(G, [
                [2 * var * A_vymax[0], 2 * var * A_vymax[1], 2 * var * A_vymax[2], 2 * var * A_vymax[3], 0, 0, 0, -1]],
                          axis=0)
            h = np.append(h, [b_staby])
        #
        #
        'Clf for r'
        #
        k_feed1 = 1
        k_feed2 = 1
        var = r_dot + k_feed1 * r + k_feed2 * ir
        V = var ** 2
        LfV = 2 * var * (k_feed1 * r_dot + k_feed2 * r + (self.Ixx - self.Iyy) * (
                p_dot * q + p * q_dot) / self.Izz + tao_psi_remain / self.Izz)
        b_stabpsi_dot = -LfV - eps * V
        A1 = -2 * w1 * self.b0 * self.b1 / self.Izz
        A2 = 2 * w2 * self.b0 * self.b1 / self.Izz
        A3 = -2 * w3 * self.b0 * self.b1 / self.Izz
        A4 = 2 * w4 * self.b0 * self.b1 / self.Izz
        A_stabpsi_dot = [2 * var * A1, 2 * var * A2, 2 * var * A3, 2 * var * A4, 0, -1, 0, 0]
        if constraintstatus[9] == 1:
            G = np.append(G, [A_stabpsi_dot], axis=0)
            h = np.append(h, [b_stabpsi_dot])
        #
        'Clf for phi'
        #
        k_feed1 = 1
        k_feed2 = 1
        var = p_dot + k_feed1 * (ip - phi_d) + k_feed2 * p
        V = var ** 2
        LfV = 2 * var * (k_feed1 * p + k_feed2 * p_dot + (self.Iyy - self.Izz) * (
                q_dot * r + q * r_dot) / self.Ixx - self.Ir * q_dot * w_t / self.Ixx - self.Ir * q * w_t_remain / self.Ixx + tao_phi_remain / self.Ixx)
        b_stabphi = -LfV - eps * V
        A_stabphi = [2 * var * A_rollmax[0], 2 * var * A_rollmax[1], 2 * var * A_rollmax[2], 2 * var * A_rollmax[3], 0, 0,
                     -1, 0]
        if constraintstatus[10] == 1:
            G = np.append(G, [A_stabphi], axis=0)
            h = np.append(h, [b_stabphi])

        'Clf for theta'
        k_feed1 = 1
        k_feed2 = 1
        var = q_dot + k_feed1 * (iq - theta_d) + k_feed2 * q
        V = var ** 2
        LfV = 2 * var * (k_feed1 * q + k_feed2 * q_dot + (self.Izz - self.Ixx) * (
                p_dot * r + p * r_dot) / self.Iyy + self.Ir * p_dot * w_t / self.Iyy + self.Ir * p * w_t_remain / self.Iyy + tao_theta_remain / self.Iyy)
        b_stabtheta = -LfV - eps * V
        A_stabtheta = [2 * var * A_pitchmax[0], 2 * var * A_pitchmax[1], 2 * var * A_pitchmax[2], 2 * var * A_pitchmax[3],
                       0, 0, 0, -1]
        if constraintstatus[11] == 1:
            G = np.append(G, [A_stabtheta], axis=0)
            h = np.append(h, [b_stabtheta])

        xo, yo, zo = [obstacle[0], obstacle[1], obstacle[2]]
        pe = 0.8
        b = (x - xo) ** 2 + (y - yo) ** 2 + (z - zo) ** 2 - 49
        b_dot = 2 * (x - xo) * x_dot + 2 * (y - yo) * y_dot + 2 * (z - zo) * z_dot
        b_ddot = 2 * x_dot ** 2 + 2 * y_dot ** 2 + 2 * z_dot ** 2 + 2 * (x - xo) * x_ddot + 2 * (y - yo) * y_ddot + 2 * (
                z - zo) * z_ddot
        Lf3b = 6 * x_dot * x_ddot + 6 * y_dot * y_ddot + 6 * z_dot * z_ddot + 2 * (x - xo) * (-Lf2bx) + 2 * (y - yo) * (
            -Lf2by) + 2 * (z - zo) * (-Lf2bz)
        A1 = 2 * (x - xo) * A_vxmin[0] + 2 * (y - yo) * A_vymin[0] + 2 * (z - zo) * A_vzmin[0]
        A2 = 2 * (x - xo) * A_vxmin[1] + 2 * (y - yo) * A_vymin[1] + 2 * (z - zo) * A_vzmin[1]
        A3 = 2 * (x - xo) * A_vxmin[2] + 2 * (y - yo) * A_vymin[2] + 2 * (z - zo) * A_vzmin[2]
        A4 = 2 * (x - xo) * A_vxmin[3] + 2 * (y - yo) * A_vymin[3] + 2 * (z - zo) * A_vzmin[3]
        b_safe = Lf3b + 3 * pe * b_ddot + 3 * pe ** 2 * b_dot + pe ** 3 * b
        A_safe = [A1, A2, A3, A4, 0, 0, 0, 0]
        if constraintstatus[12] == 1:
            G = np.append(G, [A_safe], axis=0)
            h = np.append(h, [b_safe])

        psc = 1
        H = matrix([[2, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0, 0],
                    [0, 0, 0, 0, 2 * psc, 0, 0, 0], [0, 0, 0, 0, 0, 200 * psc, 0, 0], [0, 0, 0, 0, 0, 0, 200 * psc, 0],
                    [0, 0, 0, 0, 0, 0, 0, 200 * psc]])
        F = matrix([[-2 * Fr1 / self.b1], [-2 * Fr2 / self.b1], [-2 * Fr3 / self.b1], [-2 * Fr4 / self.b1], [0], [0], [0], [0]]).trans()
        H = matrix(H, tc='d')
        F = matrix(F, tc='d')
        G = matrix(G, tc='d')
        h = matrix(h, tc='d')
        sol = qp(H, F, G, h)

        u = sol['x'].trans()
        input = []
        for i in range(len(u)): input = np.append(input, u[i])
        return input
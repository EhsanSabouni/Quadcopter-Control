import numpy as np
from scipy.integrate import odeint,solve_ivp,ode



class Quad:
    def __init__(self, M: float, Ixx: float, Iyy: float, Izz: float, Ir: float, l: float, k: float,
                 b0: float, b1: float, beta0: float, beta1: float,
                 beta2: float, k1: float, p1: float, Ax: float, Ay: float, Az: float, g: float):
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

    def dynamics(self, t,x):
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
                    np.cos(x[10]) * np.sin(x[8]) * np.cos(x[6]) - np.sin(x[10]) * np.sin(x[6])) - 1 / self.M * self.Ax * \
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

    #def motion(self, state, input, t):
        #x0 = np.append(state, input)
        #x = odeint(self.dynamics, x0, t)
        #r = ode(self.dynamics)
        #r.set_initial_value(x0, t[0])
        #t1 = t[-1]
        #dt = 0.1
        #while r.successful() and r.t < t1:
            #x = r.integrate(r.t + dt)

        #return x[-1, 0: 16]
        #return x
    def motion(self, state, input, method, timespan, teval):
        y0 = np.append(state, input)
        sol = solve_ivp(self.dynamics, timespan, y0, method=method, t_eval=[teval])
        x = sol.y

        return x

    def rk4(self, t, state, input, n):
        # Calculating step size
        #x0 = np.append(state, input)
        h = (t[-1] - t[0]) / n
        t0 = t[0]
        for i in range(n):
            k1 = (self.dynamics2(t0, state, input))
            k2 = (self.dynamics2((t0 + h / 2), (state + h * k1 / 2), input))
            k3 = (self.dynamics2((t0 + h / 2), (state + h * k2 / 2), input))
            k4 = (self.dynamics2((t0 + h), (state + h * k3), input))
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
                    np.cos(x[10]) * np.sin(x[8]) * np.cos(x[6]) - np.sin(x[10]) * np.sin(x[6])) - 1 / self.M * self.Ax * x[1])
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
    def dynamics2(self, t, x,u):
        dx = np.zeros(16)  # [(x,0), (vx,1), (y,2), (vy,3),  (z,4), (vz,5), (phi,6), (omega_phi,7), (theta,8),
        # (omega_theta,9), (psi,10), (omega_psi,11), (omega1,12), (omega2,13), (omega3,14), (omega4,15),
        #   (u1,16), (u2,17), (u3,18), (u4,19)]

        tau_phi = self.l * self.k * (-x[13] ** 2 + x[15] ** 2)
        tau_theta = self.l * self.k * (-x[12] ** 2 + x[14] ** 2)
        tau_psi = self.b0 * (-x[12] ** 2 - x[14] ** 2 + x[13] ** 2 + x[15] ** 2)
        omega_gamma = x[12] - x[13] + x[14] - x[15]
        F = self.k * (x[12] ** 2 + x[13] ** 2 + x[14] ** 2 + x[15] ** 2)

        dx[0] = x[1]
        dx[1] = F / self.M * (
                    np.cos(x[10]) * np.sin(x[8]) * np.cos(x[6]) - np.sin(x[10]) * np.sin(x[6])) - 1 / self.M * self.Ax * \
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

        dx[12] = self.b1 * u[0] - self.beta0 - self.beta1 * x[12] - self.beta2 * x[12] ** 2
        dx[13] = self.b1 * u[1] - self.beta0 - self.beta1 * x[13] - self.beta2 * x[13] ** 2
        dx[14] = self.b1 * u[2] - self.beta0 - self.beta1 * x[14] - self.beta2 * x[14] ** 2
        dx[15] = self.b1 * u[3] - self.beta0 - self.beta1 * x[15] - self.beta2 * x[15] ** 2

        return dx
import numpy as np
import sympy as sym



def dynamics(self, t, state):
    dstate = np.zeros(20)  # [(state,0), (vstate,1), (y,2), (vy,3),  (z,4), (vz,5), (phi,6), (omega_phi,7), (theta,8),
    # (omega_theta,9), (psi,10), (omega_psi,11), (omega1,12), (omega2,13), (omega3,14), (omega4,15),
    #   (u1,16), (u2,17), (u3,18), (u4,19)]

    tau_phi = self.l * self.k * (-state[13] ** 2 + state[15] ** 2)
    tau_theta = self.l * self.k * (-state[12] ** 2 + state[14] ** 2)
    tau_psi = self.b0 * (-state[12] ** 2 - state[14] ** 2 + state[13] ** 2 + state[15] ** 2)
    omega_gamma = state[12] - state[13] + state[14] - state[15]
    F = self.k * (state[12] ** 2 + state[13] ** 2 + state[14] ** 2 + state[15] ** 2)

    dstate[0] = state[1]
    dstate[1] = F / self.M * (
            np.cos(state[10]) * np.sin(state[8]) * np.cos(state[6]) - np.sin(state[10]) * np.sin(state[6])) - 1 / self.M * self.Astate * \
            state[1]
    dstate[2] = state[3]
    dstate[3] = F / self.M * (
            np.sin(state[10]) * np.sin(state[8]) * np.cos(state[6]) - np.cos(state[10]) * np.sin(state[6])) - 1 / self.M * self.Ay * \
            state[3]
    dstate[4] = state[5]
    dstate[5] = -self.g + F / self.M * (np.cos(state[8]) * np.cos(state[6])) - 1 / self.M * self.Az * state[5]

    D1 = np.array([[0, state[7] * np.cos(state[6]) * np.tan(state[8]) + state[9] * np.sin(state[6]) / (np.cos(state[8])) ** 2,
                    -state[7] * np.sin(state[6]) * np.cos(state[8]) + state[9] * np.cos(state[6]) / (np.cos(state[8])) ** 2],
                   [0, -state[7] * np.sin(state[6]), -state[7] * np.cos(state[6])],
                   [0, state[7] * np.cos(state[6]) / np.cos(state[8]) + state[7] * np.sin(state[6]) * np.tan(state[8]) / np.cos(state[8]),
                    -state[7] * np.sin(state[6]) / np.cos(state[8]) + state[9] * np.cos(state[6]) * np.tan(state[8]) / np.cos(state[8])]])

    W_eta_inv = np.array(
        [[1, np.sin(state[6]) * np.tan(state[8]), np.cos(state[6]) * np.tan(state[8])], [0, np.cos(state[6]), -np.sin(state[8])],
         [0, np.sin(state[6]) / np.cos(state[8]), np.cos(state[6]) / np.cos(state[8])]])

    W_eta = np.array([[1, 0, -np.sin(state[8])], [0, np.cos(state[6]), np.cos(state[8]) * np.sin(state[6])],
                      [0, -np.sin(state[6]), np.cos(state[8]) * np.cos(state[6])]])

    p, q, r = np.dot(W_eta, [state[7], state[9], state[11]])
    D2 = np.array(
        [(self.Iyy - self.Izz) * q * r / self.Ixx - self.Ir * (q / self.Ixx) * omega_gamma + tau_phi / self.Ixx,
         (self.Izz - self.Ixx) * p * r / self.Iyy - self.Ir * (-p / self.Iyy) * omega_gamma + tau_theta / self.Iyy,
         (self.Ixx - self.Iyy) * p * q / self.Izz + tau_psi / self.Izz])
    dstate[6] = state[7]
    dstate[7], dstate[9], dstate[11] = np.dot(D1, [p, q, r]) + np.dot(W_eta_inv, D2)

    dstate[8] = state[9]
    dstate[10] = state[11]

    dstate[12] = self.b1 * state[16] - self.beta0 - self.beta1 * state[12] - self.beta2 * state[12] ** 2
    dstate[13] = self.b1 * state[17] - self.beta0 - self.beta1 * state[13] - self.beta2 * state[13] ** 2
    dstate[14] = self.b1 * state[18] - self.beta0 - self.beta1 * state[14] - self.beta2 * state[14] ** 2
    dstate[15] = self.b1 * state[19] - self.beta0 - self.beta1 * state[15] - self.beta2 * state[15] ** 2

    dstate[16:20] = [0, 0, 0, 0]

    return dstate



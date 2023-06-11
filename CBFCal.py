import numpy as np
import sympy as sym

# x = sym.Symbol("x")
# v = sym.Symbol("v")
# u = sym.Symbol("u")
# x_ip = sym.Symbol("x_ip")
# v_ip = sym.Symbol("v_ip")
# u_ip = sym.Symbol("v_ip")
# states = [x, v, x_ip, v_ip]
# input = [u, u_ip]
# CBF = 0
# f = [v, 0, v_ip, 0]
# g = np.array([[0, 0], [1, 0], [0, 0], [0, 1]])
#
# b = x_ip - x - 1.8*v
#
# for i in range(len(states)):
#     CBF = CBF + np.dot(b.diff(states[i]), (f[i] + np.dot(g[i, :], input)))
#
# h = 1


# M = 0.468
# g = 9.81
# Ixx = 4.856 * 10 ** -3
# Iyy = 4.856 * 10 ** -3
# Izz = 8.81 * 10 ** -3
# Ir = 3.357 * 10 ** -5
# l = 0.225
# k = 2.98 * 10 ** -6
# b0 = 1.14 * 10 ** -7
b1 = 280.19
# beta0 = 189.63
# beta1 = 6.0612
# beta2 = 0.0122
# k1 = 0.5
# p1 = 5 * 10 ** -7
# Ax = 0.25
# Ay = 0.25
# Az = 0.25
#


M, g, Ixx, Iyy, Izz, Ir, l, k, b0, beta0, beta1, beta2, k1, p1, Ax, Ay, Az, \
    = sym.symbols('M g Ixx Iyy Izz Ir l k b0 beta0 beta1 beta2 k1 p1 Ax Ay Az')


x, y, z = sym.symbols('x y z')
x0, y0, z0 = sym.symbols("x0 y0 z0")
vx, vy, vz = sym.symbols("vx vy vz")
phi, theta, psi = sym.symbols("phi theta psi")
omega_phi, omega_theta, omega_psi = sym.symbols("omega_phi omega_theta omega_psi")
omega1, omega2, omega3, omega4 = sym.symbols("omega1 omega2 omega3 omega4")
u1, u2, u3, u4 = sym.symbols("u1 u2 u3 u4")
R0 = sym.symbols("R0")
k1 = sym.Symbol("k1")
k2 = sym.Symbol("k2")
k3 = sym.Symbol("k3")
 # [(x,0), (vx,1), (y,2), (vy,3),  (z,4), (vz,5), (phi,6), (omega_phi,7), (theta,8),
# (omega_theta,9), (psi,10), (omega_psi,11), (omega1,12), (omega2,13), (omega3,14), (omega4,15),
#   (u1,16), (u2,17), (u3,18), (u4,19)]


f = np.asarray(sym.symbols('f0:19'))
states = [x, vx, y, vy, z, vz, phi, omega_phi, theta, omega_theta, psi,
          omega_psi, omega1, omega2, omega3, omega4, x0, y0, z0]
inputs = [u1, u2, u3, u4]

tau_phi = l * k * (-omega2 ** 2 + omega4 ** 2)
tau_theta = l * k * (-omega1 ** 2 + omega3 ** 2)
tau_psi = b0 * (-omega1 ** 2 - omega3 ** 2 + omega2 ** 2 + omega4 ** 2)
omega_gamma = omega1 - omega2 + omega3 - omega4
F = k * (omega1 ** 2 + omega2 ** 2 + omega3 ** 2 + omega4 ** 2)

f[0] = vx
f[1] = F / M * ( sym.cos(psi) * sym.sin(theta) * sym.cos(phi) - sym.sin(psi) * sym.sin(phi)) - 1 / M * Ax * vx
f[2] = vy
f[3] = F / M * ( sym.sin(psi) * sym.sin(theta) * sym.cos(phi) - sym.cos(psi) * sym.sin(phi)) - 1 / M * Ay * vy
f[4] = vz
f[5] = -g + F / M * (sym.cos(theta) * sym.cos(phi)) - 1 / M * Az * vz

D1 = np.array([[0, omega_phi * sym.cos(phi) * sym.tan(theta) + omega_theta * sym.sin(phi) / (sym.cos(theta)) ** 2,
                -omega_phi * sym.sin(phi) * sym.cos(theta) + omega_theta * sym.cos(phi) / (sym.cos(theta)) ** 2],
               [0, -omega_phi * sym.sin(phi), -omega_phi * sym.cos(phi)],
               [0, omega_phi * sym.cos(phi) / sym.cos(theta) + omega_phi * sym.sin(phi) * sym.tan(theta) / sym.cos(theta),
                -omega_phi * sym.sin(phi) / sym.cos(theta) + omega_theta * sym.cos(phi) * sym.tan(theta) / sym.cos(theta)]])

W_eta_inv = np.array(
    [[1, sym.sin(phi) * sym.tan(theta), sym.cos(phi) * sym.tan(theta)], [0, sym.cos(phi), -sym.sin(theta)],
     [0, sym.sin(phi) / sym.cos(theta), sym.cos(phi) / sym.cos(theta)]])

W_eta = np.array([[1, 0, -sym.sin(theta)], [0, sym.cos(phi), sym.cos(theta) * sym.sin(phi)],
                  [0, -sym.sin(phi), sym.cos(theta) * sym.cos(phi)]])

p, q, r = np.dot(W_eta, [omega_phi, omega_theta, omega_psi])
D2 = np.array(
    [(Iyy - Izz) * q * r / Ixx - Ir * (q / Ixx) * omega_gamma + tau_phi / Ixx,
     (Izz - Ixx) * p * r / Iyy - Ir * (-p / Iyy) * omega_gamma + tau_theta / Iyy,
     (Ixx - Iyy) * p * q / Izz + tau_psi / Izz])
f[6] = omega_phi
f[7], f[9], f[11] = np.dot(D1, [p, q, r]) + np.dot(W_eta_inv, D2)

f[8] = omega_theta
f[10] = omega_psi

f[12] = - beta0 - beta1 * omega1 - beta2 * omega1 ** 2
f[13] = - beta0 - beta1 * omega2 - beta2 * omega2 ** 2
f[14] = - beta0 - beta1 * omega3 - beta2 * omega3 ** 2
f[15] = - beta0 - beta1 * omega4 - beta2 * omega4 ** 2
f[16] = 0
f[17] = 0
f[18] = 0
g = np.zeros((19, 4))
g[15, 3], g[14, 2], g[13, 1], g[12, 0] = [b1, b1, b1, b1]
b = (x - x0)**2 + (y - y0)**2 + (z - z0)**2 - R0**2

CBF0 = 0
for i in range(len(states)):
    CBF0 = CBF0 + np.dot(b.diff(states[i]), (f[i] + np.dot(g[i, :], inputs)))
CBF0 = CBF0 + k1 * b
CBF1 = 0
for i in range(len(states)):
    CBF1 = CBF1 + np.dot(CBF0.diff(states[i]), (f[i] + np.dot(g[i, :], inputs)))
CBF1 = CBF1 + k2 * CBF0
CBF2_lf = 0
CBF2_lg = 0
for i in range(len(states)):
    CBF2_lf = CBF2_lf + np.dot(CBF1.diff(states[i]), f[i] )
    CBF2_lg = CBF2_lg + np.dot(CBF1.diff(states[i]), np.dot(g[i, :], inputs))
CBF2_lf = CBF2_lf + k3 * CBF1
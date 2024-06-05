import numpy as np
import matplotlib.pyplot as plt

# Gegeben [SI-Units, kg, m, s]
P = np.array([10337706.5,
              48744.172,
              455.205728,
              1.13950676,
              0.00483245447,
              3.11054119e-6])

Q = np.array([1.0,
              0.00490648418,
              2.71147593e-5,
              5.42912842e-8,
              6.94958499e-12])

# Abspaltung 1
s10 = P[5] / Q[4]
s00 = (P[4] - s10 * Q[3]) / Q[4]
r30 = P[3] - s10 * Q[2] - s00 * Q[3]
r20 = P[2] - s10 * Q[1] - s00 * Q[2]
r10 = P[1] - s10 - s00 * Q[1]
r00 = P[0] - s00

# Abspaltung 2
s11 = Q[4] / r30
s01 = (Q[3] - s11 * r20) / r30
r21 = Q[2] - s11 * r10 - s01 * r20
r11 = Q[1] - s11 * r00 - s01 * r10
r01 = Q[0] - s01 * r00

# Abspaltung 3
s12 = r30 / r21
s02 = (r20 - s12 * r11) / r21
r12 = r10 - s12 * r01 - s02 * r11
r02 = r00 - s02 * r01

# Abspaltung 4
s13 = r21 / r12
s03 = (r11 - s13 * r02) / r12
r03 = r01 - s03 * r02

# Verbleibender Rest 5
s04 = r02 / r03
s14 = r12 / r03

# Matrix A und B
A = np.array([[s10,  0.0, 0.0,  0.0,  0.0],
              [0.0, -s11, 0.0,  0.0,  0.0],
              [0.0,  0.0, s12,  0.0,  0.0],
              [0.0,  0.0, 0.0, -s13,  0.0],
              [0.0,  0.0, 0.0,  0.0, s14]])

B = np.array([[s00,  1.0,  0.0,  0.0,  0.0],
              [1.0, -s01, -1.0,  0.0,  0.0],
              [0.0, -1.0,  s02,  1.0,  0.0],
              [0.0,  0.0,  1.0, -s03, -1.0],
              [0.0,  0.0,  0.0, -1.0, s04]])

# Zeitschritt
timestep = 1e-5  #[s]
t_max = 1.0      #[s]

# Belastungsfunktion (Abschnittsweise linear)
def f(t):
    F_max = 100*1000    #[N]
    l = 0.01            #[s]
    if 0 <= t < l:
        return F_max/l*t
    if l <= t < 2*l:
        return F_max
    if 2*l <= t <= 3*l:
        return F_max - F_max/l*(t-2*l)
    else:
        return 0

# Zeitabhängige Verschiebung
def z(z_current, t_current, t_new):
    inverted_part = np.linalg.inv(A + timestep/2 * B)           # matrix [5x5]

    belastung_part = np.array([[(f(t_current) + f(t_new)) / 2 * timestep],  # vector [5x1]
                               [0.0],
                               [0.0],
                               [0.0],
                               [0.0]])

    right_part = (timestep / 2 * B - A)                         # matrix [5x5]

    z_new = inverted_part @ ( belastung_part - right_part @ z_current)  # vector[5x1]
    return z_new

# Startbedingung
t_current = 0.0   # [s] Start zum Zeitpunkt t = 0
z_current = np.array([[0.0],
                      [0.0],
                      [0.0],
                      [0.0],
                      [0.0]])   # [m] keine Verschiebung

# Speicher
t_history = []
z_history = []

while t_current <= t_max:
        t_new = t_current + timestep
        t_history.append(t_current)
        z_history.append(z_current[0,0])
        z_current = z(z_current, t_current, t_new)
        t_current = t_new


# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(t_history, z_history, linestyle='-', color='gray')

# Add title and labels
plt.xlabel('Zeit [s]')
plt.ylabel('Durchbiegung v(x=0) [m]')

# Set x limit
plt.xlim(0, 1)

# Optionally, add grid
plt.grid(True)

# Show the plot
plt.show()


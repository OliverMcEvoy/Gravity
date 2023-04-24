import numpy as np
import matplotlib.pyplot as plt

# Set up constants
g = 9.81      # Acceleration due to gravity (m/s^2)
rho_a = 1.29  # Density of air (kg/m^3)
rho_w = 1000  # Density of water (kg/m^3)
rho_o = 800   # Density of oil (kg/m^3)
C_d = 0.5     # Drag coefficient
A = 0.01      # Cross-sectional area of object (m^2)
m = 0.01      # Mass of object (kg)
t_start = 0   # Start time (s)
t_end = 1     # End time (s)
dt = 0.001    # Time step (s)

# Set up arrays
t = np.arange(t_start, t_end, dt)
v_a = np.zeros_like(t)
v_w = np.zeros_like(t)
v_o = np.zeros_like(t)
y_a = np.zeros_like(t)
y_w = np.zeros_like(t)
y_o = np.zeros_like(t)

# Calculate velocities and positions
for i in range(1, len(t)):
    # Air
    F_a = 0.5 * rho_a * C_d * A * v_a[i-1]**2
    a_a = -g + F_a / m
    v_a[i] = v_a[i-1] + a_a * dt
    y_a[i] = y_a[i-1] + v_a[i] * dt

    # Water
    F_w = 0.5 * rho_w * C_d * A * v_w[i-1]**2
    a_w = -g + F_w / m
    v_w[i] = v_w[i-1] + a_w * dt
    y_w[i] = y_w[i-1] + v_w[i] * dt

    # Oil
    F_o = 0.5 * rho_o * C_d * A * v_o[i-1]**2
    a_o = -g + F_o / m
    v_o[i] = v_o[i-1] + a_o * dt
    y_o[i] = y_o[i-1] + v_o[i] * dt

# Plot results
fig, ax = plt.subplots()
ax.plot(t, y_a, label='Air')
ax.plot(t, y_w, label='Water')
ax.plot(t, y_o, label='Oil')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Height (m)')
ax.legend()
plt.show()

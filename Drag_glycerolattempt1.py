import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import curve_fit

Exp_1 = pd.read_csv('Data/drag_data.csv')


g = 9.81      # Acceleration due to gravity (m/s^2)
rho_a = 1.29  # Density of air (kg/m^3)
rho_w = 1261.3   # Density of water (kg/m^3)
rho_o = 1261.3   # Density of oil (kg/m^3)
C_d = 0.45    # Drag coefficient
viscosity = 1.412
radius = 0.01794/2  # radius in m for the big bal
A = np.pi*radius**2      # Cross-sectional area of object (m^2)
# m = 0.02377      # Mass of object (kg)
m = 8000*(4/3*np.pi*(radius**3))
print(m)
t_start = 0   # Start time (s)
t_end = 1.4    # End time (s)
dt = 0.001    # Time step (s)
Decimal = 10
exponent = 2

# Exp_1['x_m']=Exp_1['x_m']*10-0.2

e = 1
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
    a_w = g - F_w / m
    v_w[i] = v_w[i-1] + a_w * dt
    y_w[i] = y_w[i-1] + v_w[i] * dt

    # Oil
    F_o = 6 * np.pi * radius * v_o[i-1] * viscosity
    F_o_d = 0.5 * rho_o * C_d * A * v_o[i-1]**exponent
    a_o = g - F_o / m
    v_o[i] = v_o[i-1] + a_o * dt
    if ((str(v_o[i])[:Decimal] == str(v_o[i-1])[:Decimal]) and (e == 1)):
        print("\n The terminal velocity of oil is =",
              v_o[i], "m/s Occuring at", i*dt, "seconds")
        e = e+1
    y_o[i] = y_o[i-1] + v_o[i] * dt


def line(x, slope, intercept):          # Set up the linear fitting - don't ammend
    return slope*x + intercept          # More set up, leave alone.

# Next few line, fits a line to the (x data, and y data) no need to change things.


def line(x, slope, intercept):          # Set up the linear fitting - don't ammend
    return slope*x + intercept


def curve(time, a, b, c):
    return a*time**2+b*time+c


# Next few line, fits a line to the (x data, and y data) no need to change things.
popt, pcov = curve_fit(curve, Exp_1['t_avg'], Exp_1['x_m'])
slope = popt[0]
slope2 = popt[1]
intercept = popt[2]
err_slope = np.sqrt(float(pcov[0][0]))
err_slope2 = np.sqrt(float(pcov[1][1]))
err_intercept = np.sqrt(float(pcov[2][2]))
# print(slope)
# print(err_slope)

print("g is ", slope*2)
print("with error", err_slope*2)

print("gradient is ", slope2)
print("with error", err_slope2*2)

print("intercept is ", intercept)
print("with error", err_intercept)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)


ax.errorbar(
    t,
    slope*t**2+slope2*t+intercept,  # /(np.cos(x-90)),
    #                xerr=0.1,
    #            yerr=0.1,     # y errors
    # marker used is a cicle 'o'. Could be crosses 'x', or squares 's', or 'none'
    marker='x',
    markersize=0,        # marker size
    color='purple',          # overall colour I think
    #             ecolor='black',         # edge colour for you marker
    #             markerfacecolor='black',
    # no line joining markers, could be a line '-', or a dashed line '--'
    linestyle='solid',
    # width of the end bit of the error bars, not too small nor too big please.
    capsize=4,
    label='Fitted line',
    linewidth=1,
    alpha=1
)

ax.errorbar(
    t,
    y_o,  # /(np.cos(x-90)),
    #                xerr=0.1,
    #            yerr=0.1,     # y errors
    # marker used is a cicle 'o'. Could be crosses 'x', or squares 's', or 'none'
    marker='x',
    markersize=0,        # marker size
    #             color='black',          # overall colour I think
    #             ecolor='black',         # edge colour for you marker
    #             markerfacecolor='black',
    linestyle='--',       # no line joining markers, could be a line '-', or a dashed line '--'
    # width of the end bit of the error bars, not too small nor too big please.
    capsize=4,
    label='Simulated Model',
    linewidth=1,
    alpha=1
)

ax.errorbar(
    t,
    y_w,  # /(np.cos(x-90)),
    #                xerr=0.1,
    #            yerr=0.1,     # y errors
    # marker used is a cicle 'o'. Could be crosses 'x', or squares 's', or 'none'
    marker='x',
    markersize=0,        # marker size
    #             color='black',          # overall colour I think
    #             ecolor='black',         # edge colour for you marker
    #             markerfacecolor='black',
    linestyle='--',       # no line joining markers, could be a line '-', or a dashed line '--'
    # width of the end bit of the error bars, not too small nor too big please.
    capsize=4,
    label='Simulated Model',
    linewidth=1,
    alpha=1
)


ax.errorbar(
    Exp_1['t_avg'],
    Exp_1['x_m'],  # /(np.cos(x-90)),
    xerr=0.005,
    yerr=0.01,     # y errors
    # marker used is a cicle 'o'. Could be crosses 'x', or squares 's', or 'none'
    marker='o',
    markersize=4,        # marker size
    color='black',          # overall colour I think
    #             ecolor='black',         # edge colour for you marker
    markerfacecolor='orange',
    # no line joining markers, could be a line '-', or a dashed line '--'
    linestyle='none',
    # width of the end bit of the error bars, not too small nor too big please.
    capsize=3,
    label='Experimental',
    linewidth=1,
    alpha=1
)


# You can use Latex here is you wish, e.g., 'Distance run / 10$^{-3}$ km'
plt.xlabel('Time taken to fall s')
plt.ylabel('Displacement m ')

# plt.xlim(35, 60)                    # extent of the x axis, smallest to largest.
# plt.ylim(-1.5, 10.5)

plt.tick_params(direction='in',      # I like 'in', could be 'out' or both 'inout'
                length=7,            # A reasonable length
                bottom='on',         # I want ticks on the bottom axes
                left='on',
                top='on',
                right='on',

                )

# A decent font size so the text is readible.
plt.rcParams.update({'font.size': 15})
# save the graph to a file.
# You may have to play with the aspect ration aobe and this to get a nice
# looking figure in your report.
plt.legend(loc="upper left")
fig.savefig('resultsfalling.png', dpi=300)
# ax.set_ylim(-0.1, 1)
plt.show()  # Display the graph below.

'''results=Exp_1['suvat'].iloc[:6]
print(results)

avg=sum(results)/len(results)
std=np.std(results)
print(avg)
print(std)'''

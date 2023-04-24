# Import necessary libraries
from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

# Define the function for the derivative of the state vector
# state[0] is the angle, state[1] is the angular velocity


def derivs(state, t):
    # create an array of zeros of the same shape as state
    dydx = np.zeros_like(state)
    # the derivative of the angle is the angular velocity
    dydx[0] = state[1]
    # the derivative of the angular velocity is the negative of (g/L)*sin(angle)
    dydx[1] = -(g/L) * sin(state[0])
    return dydx


i = 1

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)

while (i < 10):
    # Set the acceleration due to gravity (m/s^2) and the length of the pendulum (m)
    g = 9.8
    L = 1.2
# Set the initial angle (in degrees) and angular velocity (in degrees per second)
    theta0 = (20*i)-1
    w0 = 0
# Set the mass of the pendulum in kg (not needed yet)
    m = 1


# Create a time array with a specified time step (dt)
    dt = 0.1  # lower value better but my laptop does not like
    t = np.arange(0.0, 100, dt)


# Convert the initial angle and angular velocity to radians and create the initial state vector
    state = np.radians([theta0, w0])

# Integrate the state vector using odeint to get the angle and angular velocity as functions of time
    z = integrate.odeint(derivs, state, t)


# Calculate the x and y positions of the pendulum bob based on the angle and length of the pendulum
    x = L * sin(z[:, 0])
    y = -L * cos(z[:, 0])

# Set up the figure and axes for the animation

    ax.errorbar(
        z[:, 0],  # /(np.cos(x-90)),
        z[:, 1],

        #                xerr=0.1,
        #             yerr=0.1,
        # marker used is a cicle 'o'. Could be crosses 'x', or squares 's', or 'none'
        marker='X',
        markersize=0,        # marker size
        color='black',          # overall colour I think
        ecolor='black',         # edge colour for you marker
        markerfacecolor='black',
        # no line joining markers, could be a line '-', or a dashed line '--'
        linestyle='solid',
        # width of the end bit of the error bars, not too small nor too big please.
        capsize=4,
        label='predicted',
        linewidth=1,
        alpha=1
    )
    i = i+1


# You can use Latex here is you wish, e.g., 'Distance run / 10$^{-3}$ km'
plt.xlabel('Angle')
plt.ylabel('Angular Velcotiy')

# plt.xlim(35, 60)                    # extent of the x axis, smallest to largest.
# plt.ylim(-1.5, 10.5)

plt.show()

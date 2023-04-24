# Import necessary libraries
from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
plt.rcParams.update({'font.size': 18})


# Set the acceleration due to gravity (m/s^2) and the length of the pendulum (m)
g = 9.8
L = 1.2

theta_init = 16
# Set the initial angle (in $^o$) and angular velocity (in $^o$ per second)
theta0 = theta_init
w0 = 0.0
# Set the mass of the pendulum in kg (not needed yet)
m = 1
i = 0


# Define the function for the derivative of the state vector
# state[0] is the angle, state[1] is the angular velocity
def derivs(state, t):
    # create an array of zeros of the same shape as state
    dydx = np.zeros_like(state)
    # the derivative of the angle is tx    e angular velocity
    dydx[0] = state[1]
    # the derivative of the angular velocity is the negative of (g/L)*sin(angle)
    dydx[1] = -(g/L) * sin(state[0])
    return dydx


# Create a time array with a specified time step (dt)
dt = 0.01  # lower value better but my laptop does not like
t = np.arange(0.0, 100, dt)

fig = plt.figure()
ax2 = fig.add_subplot(1, 2, 2)
ax = fig.add_subplot(1, 2, 1)

plt.tick_params(direction='in',      # I like 'in', could be 'out' or both 'inout'
                length=7,            # A reasonable length
                bottom='on',         # I want ticks on the bottom axes
                left='on',
                top='on',
                right='on',

                )

# Define the initialization function for the animation


def init():
    line.set_data([], [])  # clear the data from the line object
    time_text.set_text('')  # clear the text from the time object
    return line, time_text

# Define the animation function for each frame


def animate(i):
    # Set the x and y positions of the line object based on the current angle and length of the pendulum
    thisx = [0, x[i]]
    thisy = [0, y[i]]

    line.set_data(thisx, thisy)  # set the data for the line object
    time_text.set_text(time_template % (i*dt))  # update the time text object
    return line, time_text


# Create the animation using FuncAnimation and save it as an MP4 file


while (i < 8):

    # Create a time array with a specified time step (dt)
    theta0 = 10*1.5**i

# Convert the initial angle and angular velocity to radians and create the initial state vector
    state = np.radians([theta0, w0])

# Integrate the state vector using odeint to get the angle and angular velocity as functions of time
    z = integrate.odeint(derivs, state, t)


# Calculate the x and y positions of the pendulum bob based on the angle and length of the pendulum
    x = L * sin(z[:, 0])
    y = -L * cos(z[:, 0])

# Set up the figure and axes for the animation

    ax.errorbar(
        z[:, 0]/np.pi*180,  # /(np.cos(x-90)),
        z[:, 1],

        #                xerr=0.1,
        #             yerr=0.1,
        # marker used is a cicle 'o'. Could be crosses 'x', or squares 's', or 'none'
        marker='o',
        markersize=1,        # marker size
        color='grey',          # overall colour I think
        ecolor='black',         # edge colour for you marker
        markerfacecolor='none',
        # no line joining markers, could be a line '-', or a dashed line '--'
        linestyle='--',
        # width of the end bit of the error bars, not too small nor too big please.
        capsize=4,
        label='predicted',
        linewidth=1,
        alpha=1
    )
    i = i+1
theta0 = theta_init

# Convert the initial angle and angular velocity to radians and create the initial state vector
state = np.radians([theta0, w0])

# Integrate the state vector using odeint to get the angle and angular velocity as functions of time
z = integrate.odeint(derivs, state, t)

# Calculate the x and y positions of the pendulum bob based on the angle and length of the pendulum
x = L * sin(z[:, 0])/np.pi*180
y = -L * cos(z[:, 0])

# Set up the figure and axes for the animation


# Create an empty line object to represent the pendulum bob
line, = ax2.plot([], [], 'o-', lw=1, color='blue')

# Create a text object to display the current time
time_template = 'time = %.1fs'
time_text = ax2.text(0.05, 0.9, '', transform=ax.transAxes)


# You can use Latex here is you wish, e.g., 'Distance run / 10$^{-3}$ km'
plt.xlabel('Angle from origin ($^o$)')
plt.ylabel('Angular velocity ($^o$ s$^-1$)')

ax.errorbar(
    z[:, 0]/np.pi*180,  # /(np.cos(x-90)),
    z[:, 1],

    #                xerr=0.1,
    #             yerr=0.1,
    # marker used is a cicle 'o'. Could be crosses 'x', or squares 's', or 'none'
    marker='o',
    markersize=0,        # marker size
    color='blue',          # overall colour I think
    ecolor='black',         # edge colour for you marker
    markerfacecolor='black',
    # no line joining markers, could be a line '-', or a dashed line '--'
    linestyle='solid',
    # width of the end bit of the error bars, not too small nor too big please.
    capsize=4,
    label='predicted',
    linewidth=2,
    alpha=1
)

# plt.xlim(35, 60)                    # extent of the x axis, smallest to largest.
# plt.ylim(-1.5, 10.5)

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
                              interval=1000*dt, blit=True, init_func=init)

plt.tick_params(direction='in',      # I like 'in', could be 'out' or both 'inout'
                length=7,            # A reasonable length
                bottom='on',         # I want ticks on the bottom axes
                left='on',
                top='on',
                right='on',

                )

ax2.set_xlim([-60, 60])
# plt.ylim([-1.5,0.5])
ax2.set_ylim([-1.5, 0.5])
ax2.set(ylabel=None)
ax2.set_yticks([])
ax2.set_xlabel('Angle from origin ($^o$)')


# ani.save('single_pendulum.mp4', fps=15)


plt.show()

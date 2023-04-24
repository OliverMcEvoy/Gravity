
# Update positions of labels
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set up constants
GRAVITY_ACCELERATION = 9.81  # Acceleration due to gravity (m/s^2)
AIR_DENSITY = 1.29           # Density of air (kg/m^3)
WATER_DENSITY = 1000         # Density of water (kg/m^3)
OIL_DENSITY = 1200           # Density of oil (kg/m^3)
DRAG_COEFFICIENT = 0.45       # Drag coefficient
CROSS_SECTIONAL_AREA = 0.01  # Cross-sectional area of object (m^2)
MASS = 1                     # Mass of object (kg)
START_TIME = 0               # Start time (s)
END_TIME = 5                 # End time (s)
TIME_STEP = 0.01          # Time step (s)
Decimal = 5            # desired accuracy

e = 1
f = 1
g = 1
# Set up arrays
time_array = np.arange(START_TIME, END_TIME, TIME_STEP)
air_velocity_array = np.zeros_like(time_array)
water_velocity_array = np.zeros_like(time_array)
oil_velocity_array = np.zeros_like(time_array)
air_position_array = np.zeros_like(time_array)
water_position_array = np.zeros_like(time_array)
oil_position_array = np.zeros_like(time_array)

# Calculate velocities and positions
for i in range(1, len(time_array)):
    # Air
    air_drag_force = 0.5 * AIR_DENSITY * DRAG_COEFFICIENT * \
        CROSS_SECTIONAL_AREA * air_velocity_array[i-1]**2
    air_acceleration = -GRAVITY_ACCELERATION + air_drag_force / MASS
    air_velocity_array[i] = air_velocity_array[i-1] + \
        air_acceleration * TIME_STEP
    if ((str(air_velocity_array[i]-air_velocity_array[i-1])[:Decimal] == 0) and (g == 1)):
        print("\n The terminal velocity of Air is =",
              air_velocity_array[i], "m/s Occuring at", i*TIME_STEP, "seconds")
        g = g+1
    air_position_array[i] = air_position_array[i-1] + \
        air_velocity_array[i] * TIME_STEP

    # Water
    water_drag_force = 0.5 * WATER_DENSITY * DRAG_COEFFICIENT * \
        CROSS_SECTIONAL_AREA * water_velocity_array[i-1]**2
    water_acceleration = -GRAVITY_ACCELERATION + water_drag_force / MASS
    water_velocity_array[i] = water_velocity_array[i-1] + \
        water_acceleration * TIME_STEP
    if ((str(water_velocity_array[i])[:Decimal] == str(water_velocity_array[i-1])[:Decimal]) and (f == 1)):
        print("\n The terminal velocity of Water is =",
              air_velocity_array[i], "m/s Occuring at", i*TIME_STEP, "seconds")
        f = f+1
    water_position_array[i] = water_position_array[i-1] + \
        water_velocity_array[i] * TIME_STEP

    # Oil
    oil_drag_force = 0.5 * OIL_DENSITY * DRAG_COEFFICIENT * \
        CROSS_SECTIONAL_AREA * oil_velocity_array[i-1]**2
    oil_acceleration = -GRAVITY_ACCELERATION + oil_drag_force / MASS
    oil_velocity_array[i] = oil_velocity_array[i-1] + \
        oil_acceleration * TIME_STEP
    if ((str(oil_velocity_array[i])[:Decimal] == str(oil_velocity_array[i-1])[:Decimal]) and (e == 1)):
        print("\n The terminal velocity of oil is =",
              oil_velocity_array[i], "m/s Occuring at", i*TIME_STEP, "seconds")
        e = e+1
    oil_position_array[i] = oil_position_array[i-1] + \
        oil_velocity_array[i] * TIME_STEP

# Set up figure and axis
figure, axis = plt.subplots(figsize=(4, 4))
axis.set_xlim(-3, 3)
axis.set_ylim(-30, 0)
axis.set_xlabel('Horizontal position (m)')
axis.set_ylabel('Vertical position (m)')

# Initialize points
air_point, = axis.plot([], [], 'o', color='blue')
water_point, = axis.plot([], [], 'o', color='green')
oil_point, = axis.plot([], [], 'o', color='red')

# Add labels for each point
air_label = axis.text(0, 0, 'Air', ha='center', va='bottom', color='blue')
water_label = axis.text(0, 0, 'Water', ha='center', va='bottom', color='green')
oil_label = axis.text(0, 0, 'Oil', ha='center', va='bottom', color='red')


# Initialize text for simulated time
simulated_time_text = axis.text(0.1, 0.95, '', transform=axis.transAxes)


def update(frame):
  # Update positions of points
    air_point.set_data(-1, air_position_array[frame])
    water_point.set_data(0, water_position_array[frame])
    oil_point.set_data(1, oil_position_array[frame])

    # Update positions of labels
    # i dont know why double brackes are needed but for some reason only works with them added
    air_label.set_position((-1, air_position_array[frame]))
    water_label.set_position((0, water_position_array[frame]))
    oil_label.set_position((1, oil_position_array[frame]))

    # Update simulated time text
    simulated_time_text.set_text('Time: {:.2f} s'.format(frame * TIME_STEP))

    # Return points and text
    return air_point, water_point, oil_point, air_label, water_label, oil_label, simulated_time_text


animation = animation.FuncAnimation(figure, update, frames=len(
    time_array), interval=TIME_STEP*1000, blit=True)
plt.show()

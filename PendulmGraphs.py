import matplotlib.pyplot as plt  # plotting things
import numpy as np  # one of python's main maths packages
import pandas as pd  # for reading in our data
from scipy.optimize import curve_fit  # for fitting a line to our data
# this one lets us change some parameters in our plots.
import matplotlib.ticker as ticker
import math


def quadratic_formula(a, b, c):
    # Calculate the discriminant
    discriminant = b**2 - 4*a*c

    # Check if the discriminant is negative (no real roots)
    if discriminant < 0:
        return None

    # Calculate the two roots
    root1 = (-b + math.sqrt(discriminant)) / (2*a)
    root2 = (-b - math.sqrt(discriminant)) / (2*a)

    # Return the roots
    return (root1, root2)

def calculate_g(T1, T2, l1, l2):
    numerator = 8 * np.pi ** 2
    denominator = ((T1 ** 2 + T2 ** 2) / (l1 + l2)) + ((T1 ** 2 - T2 ** 2) / (l1 - l2))
    g = numerator / denominator
    return g

def quadratic(distance, a, b, c):
    return (a*(distance**2)) + (b*distance) + c


Data = pd.read_csv("Data/pendulum_data_2_copy.csv") # a copy of the data file with the first 3 lines missing

Data['T1_error'] = ((Data['m1_T2_s']-Data['m1_T1_s'])**2)**(1/2)


popt, pcov = curve_fit(
    quadratic, Data['distance_m'], Data['m1_T_avg'], sigma=Data['T1_error'], absolute_sigma=True, )

Fit_a_1 = popt[0]
Fit_b_1 = popt[1]
Fit_c_1 = popt[2]


err_a_1 = np.sqrt(float(pcov[0][0]))
err_b_1 = np.sqrt(float(pcov[1][1]))
err_c_1 = np.sqrt(float(pcov[2][2]))


x_1 = np.arange(0, 0.85, 0.01)
y_1 = (Fit_a_1*(x_1**2)) + (Fit_b_1*x_1) + Fit_c_1

Data = pd.read_csv("Data/pendulum_data_2.csv")

Data['T1_error'] = ((Data['m1_T2_s']-Data['m1_T1_s'])**2)**(1/2)

# T2

Data['T2_error'] = ((Data['m2_T2_s']-Data['m2_T1_s'])**2)**(1/2)

popt, pcov = curve_fit(
    quadratic, Data['distance_m'], Data['m2_T_avg'], sigma=Data['T1_error'], absolute_sigma=True, )

Fit_a_2 = popt[0]
Fit_b_2 = popt[1]
Fit_c_2 = popt[2]

err_a_2 = np.sqrt(float(pcov[0][0]))
err_b_2 = np.sqrt(float(pcov[1][1]))
err_c_2 = np.sqrt(float(pcov[2][2]))


x_2 = np.arange(0, 0.85, 0.01)
y_2 = (Fit_a_2*(x_2**2)) + (Fit_b_2*x_2) + Fit_c_2


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)

ax.errorbar(
    x_1,
    y_1,  # /(np.cos(x-90)),
    #                xerr=0.1,
    #             yerr=0.1,     # y errors
    # marker used is a cicle 'o'. Could be crosses 'x', or squares 's', or 'none'
    marker='x',
    markersize=0,        # marker size
    color='blue',          # overall colour I think
    ecolor='black',         # edge colour for you marker
    #            markerfacecolor='black',
    linestyle='dashed',       # no line joining markers, could be a line '-', or a dashed line '--'
    # width of the end bit of the error bars, not too small nor too big please.
    capsize=4,
    label='Pedulum 1',
    linewidth=1,
    alpha=1
)


ax.errorbar(
    x_2,
    y_2,  # /(np.cos(x-90)),
    #                xerr=0.1,
    #             yerr=0.1,     # y errors
    # marker used is a cicle 'o'. Could be crosses 'x', or squares 's', or 'none'
    marker='x',
    markersize=0,        # marker size
    color='red',          # overall colour I think
    ecolor='black',         # edge colour for you marker
    markerfacecolor='black',
    linestyle='dotted',       # no line joining markers, could be a line '-', or a dashed line '--'
    # width of the end bit of the error bars, not too small nor too big please.
    capsize=4,
    label='Pendulum 2',
    linewidth=2,
    alpha=1
)

ax.errorbar(
    Data['distance_m'],
    Data['m1_T_avg'],
    xerr=0.01,
    yerr=Data['T1_error'],     # y errors
    # marker used is a cicle 'o'. Could be crosses 'x', or squares 's', or 'none'
    marker='o',
    markersize=4,        # marker size
           color='black',          # overall colour I think
    #           ecolor='black',         # edge colour for you marker
    markerfacecolor='blue',
    # no line joining markers, could be a line '-', or a dashed line '--'
    linestyle='none',
    # width of the end bit of the error bars, not too small nor too big please.
    capsize=2,
    label='theoretical',
    linewidth=1,
    alpha=1
)

ax.errorbar(
    Data['distance_m'],
    Data['m2_T_avg'],
    xerr=0.01,
    yerr=Data['T2_error'],     # y errors
    # marker used is a cicle 'o'. Could be crosses 'x', or squares 's', or 'none'
    marker='^',
    markersize=6,        # marker size
    color='black',          # overall colour I think
    #             ecolor='black',         # edge colour for you marker
    markerfacecolor='pink',
    # no line joining markers, could be a line '-', or a dashed line '--'
    linestyle='none',
    # width of the end bit of the error bars, not too small nor too big please.
    capsize=2,
    label='theoretical',
    linewidth=1,
    alpha=1
)

# You can use Latex here is you wish, e.g., 'Distance run / 10$^{-3}$ km'
plt.xlabel('Distance of $m_2$ form Pivot')
plt.ylabel('Period of 30 oscillations')

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
plt.tight_layout()
fig.savefig('CompoundPendulum.png', dpi=300)

 # Display the graph below.


# finding the intercepts

a = Fit_a_1-Fit_a_2
b = Fit_b_1-Fit_b_2
c = Fit_c_1-Fit_c_2

x_intercept = quadratic_formula(a, b, c)
print(x_intercept)

y_intercept_1 = (Fit_a_2*(x_intercept[0]**2)) + \
    (Fit_b_2*x_intercept[0]) + Fit_c_2
print(y_intercept_1)

y_intercept_1_error =y_intercept_1*(((err_a_2/Fit_a_2)*(x_intercept[0]**2))**2 + ((err_b_2/Fit_b_2)*x_intercept[0])**2 + (err_c_2/Fit_c_2)**2)*(1/2)
#print(y_intercept_1_error)


g1 = (2*np.pi)**2*(0.9939/(y_intercept_1/30)**2)
print("Value for g1",g1)

print(x_intercept[1])
y_intercept_2 = (Fit_a_2*(x_intercept[1]**2)) + \
    (Fit_b_2*x_intercept[1]) + Fit_c_2
print(y_intercept_2)

g2 = (2*np.pi)**2*(0.9939/(y_intercept_2/30)**2)
print("Print value for ",g2)

#Katers Pendulum

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)

Data=Data.dropna(subset=['l1'])
Data['Kater_g']=calculate_g(Data['m1_T_avg']/30,Data['m2_T_avg']/30,Data['l1'],Data['l2'])
Data['Kater_g_error']=calculate_g(Data['m1_T_avg']/30+Data['T1_error']/30,Data['m2_T_avg']/30+Data['T2_error']/30,Data['l1'],Data['l2'])
#print(Data['Kater_g'])

ax.errorbar(
    Data['distance_m'],
    Data['Kater_g'],  # /(np.cos(x-90)),
    #                xerr=0.1,
                yerr=abs(Data['Kater_g_error']-Data['Kater_g']),     # y errors
    # marker used is a cicle 'o'. Could be crosses 'x', or squares 's', or 'none'
    marker='o',
    markersize=4,        # marker size
    color='black',          # overall colour I think
#    ecolor='black',         # edge colour for you marker
                markerfacecolor='orange',
    linestyle='none',       # no line joining markers, could be a line '-', or a dashed line '--'
    # width of the end bit of the error bars, not too small nor too big please.
    capsize=4,
    label='Pedulum 1',
    linewidth=1,
    alpha=1
)
fig.savefig('Kater.png')

plt.show()

refined=Data['Kater_g'].iloc[:12]
#print(refined)
avg=sum(refined)/len(refined)
print(avg)
print(np.std(refined))
import matplotlib.pyplot as plt
import numpy as np
from SIDHE import SIDHE

#Import real datasets
Dtrue = np.genfromtxt('COVID19-US.csv', delimiter=',')
Etrue =  np.genfromtxt('COVID19-US_Deaths.csv', delimiter=',')
Htrue = np.genfromtxt('COVID19-US_Recovered.csv', delimiter=',')
#Read parameters.txt and pass to SIDHE model
with open('parametersfull.txt', 'r') as f:
    alpha, beta, gamma, delta, zeta, omega = map(float, f.read().split(','))
model = SIDHE(alpha, beta, gamma, delta, zeta, omega, 1000)

#Run both simulations
S,I,D,H,E = model.fullsimulation()[0:5]
Dtrue = np.divide(Dtrue, model.N)
Htrue = np.divide(Htrue, model.N)
Etrue = np.divide(Etrue, model.N)

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

fig, ax = plt.subplots(3, 1)

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

ax[0].plot(Dtrue, label = 'Data', color='r')
ax[0].plot(D, label = 'Model', color='y')
ax[0].legend()
ax[0].set_title("Diagnoses", fontsize=MEDIUM_SIZE)
ax[0].set_ylabel("Proportion of Population", fontsize=MEDIUM_SIZE)
ax[0].set_xlabel("Time", fontsize=MEDIUM_SIZE)

ax[1].plot(Etrue, label = 'Data', color='r')
ax[1].plot(E, label = 'Model', color='k')
ax[1].legend()
ax[1].set_title("Deaths", fontsize=MEDIUM_SIZE)
ax[1].set_ylabel("Proportion of Population", fontsize=MEDIUM_SIZE)
ax[1].set_xlabel("Time", fontsize=MEDIUM_SIZE)

ax[2].plot(Htrue, label = 'Data', color='r')
ax[2].plot(H, label = 'Model', color='c')
ax[2].legend()
ax[2].set_title("Recoveries", fontsize=MEDIUM_SIZE)
ax[2].set_ylabel("Proportion of Population", fontsize=MEDIUM_SIZE)
ax[2].set_xlabel("Time", fontsize=MEDIUM_SIZE)

fig.suptitle('Comparison Between COVID-19 Data and the SIDHE Model', fontsize=BIGGER_SIZE)

plt.figure(2)
plt.plot(S, label = 'Susceptible', color='m')
plt.plot(I, label = 'Infected', color='g')
plt.plot(D, label = 'Diagnosed', color='y')
plt.plot(H, label = 'Healed', color='c')
plt.plot(E, label = 'Expired', color='k')
plt.legend()
plt.title("Time Evolution of the SIDHE Model for COVID-19", fontsize=BIGGER_SIZE)
plt.ylabel("Proportion of Population")
plt.xlabel("Time")

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.show()
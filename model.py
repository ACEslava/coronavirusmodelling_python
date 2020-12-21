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

plt.figure(2)
plt.plot(Dtrue, label = 'Data', color='r')
plt.plot(D, label = 'Model', color='y')
plt.legend()

plt.figure(3)
plt.plot(Etrue, label = 'Data', color='r')
plt.plot(E, label = 'Model', color='k')
plt.legend()

plt.figure(4)
plt.plot(Htrue, label = 'Data', color='r')
plt.plot(H, label = 'Model', color='c')
plt.legend()

plt.figure(5)
plt.plot(S, label = 'Susceptible', color='m')
plt.plot(I, label = 'Infected', color='g')
plt.plot(D, label = 'Diagnosed', color='y')
plt.plot(H, label = 'Healed', color='c')
plt.plot(E, label = 'Expired', color='k')
plt.legend()

plt.show()
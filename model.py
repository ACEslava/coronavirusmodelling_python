import matplotlib.pyplot as plt
import numpy as np
from SIDHE import SIDHE

#Import real datasets
Dtrue = np.genfromtxt('COVID19-US.csv', delimiter=',')
Etrue =  np.genfromtxt('COVID19-US_Deaths.csv', delimiter=',')
Htrue = np.genfromtxt('COVID19-US_Recovered.csv', delimiter=',')
time = len(Dtrue)

#Read parameters.txt and pass to SIDHE model
with open('parameters.txt', 'r') as savefile:
    parameters = savefile.read().split(',')
    parameters = [float(i) for i in parameters]
model = SIDHE(*parameters, time)

#Run both simulations
model.time = 1000
Snet,Inet,Dnet,Hnet,Enet = model.fullsimulation()[0:5]
Dtrue = np.divide(Dtrue, model.N)
Htrue = np.divide(Htrue, model.N)
Etrue = np.divide(Etrue, model.N)
plt.figure(2)
plt.plot(Dtrue)
plt.plot(Dnet)

plt.figure(3)
plt.plot(Etrue)
plt.plot(Enet)

plt.figure(4)
plt.plot(Htrue)
plt.plot(Hnet)

plt.figure(5)
plt.plot(Snet, label = 'Susceptible')
plt.plot(Inet, label = 'Infected')
plt.plot(Dnet, label = 'Diagnosed')
plt.plot(Hnet, label = 'Healed')
plt.plot(Enet, label = 'Expired')
plt.legend()

plt.show()
import matplotlib.pyplot as plt
import numpy as np
from SIDHE import SIDHE

#Import real datasets
with open('parametersfull.txt', 'r') as f:
    alpha, beta, gamma, delta, zeta, omega = map(float, f.read().split(','))

#Initial parameter guess
fullmodel = SIDHE(alpha, beta, gamma, delta, zeta, omega, 1000)
ModelError = []
t = 1

#Initial Error
S, I, D, H, E = fullmodel.simplesimulation()[0:5] #Runs SIDHE to find initial error
Dtrue, Htrue, Etrue = fullmodel.fullsimulation()[2:5]
Dtrue = np.divide(Dtrue, fullmodel.N) #Makes values relative to size of population
Htrue = np.divide(Htrue, fullmodel.N)
Etrue = np.divide(Etrue, fullmodel.N)
ModelError.append( #Calculates error
    fullmodel.error(Dtrue, D) + 
    fullmodel.error(Htrue, H) + 
    fullmodel.error(Etrue, E)) 
ModelError.append(ModelError[0])
ErrorDelta = []
ErrorDelta.append(0)

while ModelError[t] <= ModelError[t-1] and t < 1000:
    #Tests error of each iteration
    S, I, D, H, E = fullmodel.simplesimulation()[0:5]
    ModelError.append( #Calculates error
        fullmodel.error(Dtrue, D) + 
        fullmodel.error(Htrue, H) + 
        fullmodel.error(Etrue, E))

    #Changes parameters by small amount
    ChangeD = [0]*6
    ChangeH = [0]*6
    ChangeE = [0]*6
    for element, letter in enumerate('abgdzo'): #Fullsimulation method checks for a char in 'abgdzo' to know which param. to change
        ChangeD[element], ChangeH[element], ChangeE[element] = fullmodel.simplesimulation(letter)[2:5] #Runs the simulation w/ changes

    #Finds derivative of the error function
    DerModel = [0]*6
    for i in range(6):
        DerModel[i] = (
            fullmodel.derivative(Dtrue, D, ChangeD[i]) + 
            fullmodel.derivative(Htrue, H, ChangeH[i]) + 
            fullmodel.derivative(Etrue, E, ChangeE[i]))

    #Gradient Descent
    h = 1E-9

    fullmodel.alpha -= h*DerModel[0]
    fullmodel.beta -= h*DerModel[1]
    fullmodel.gamma -= h*DerModel[2]
    fullmodel.delta -= h*DerModel[3]
    fullmodel.zeta -= h*DerModel[4]
    fullmodel.omega -= h*DerModel[5]
    
    #Print status of learning 
    ErrorDelta.append(ModelError[t] - ModelError[t-1])
    print(f'Iteration {t:4d} |'
        f' Error: {ModelError[t]:.20f} |' 
        f' Delta: {(ModelError[t] - ModelError[t-1]):.20f}' 
    )
    t += 1

ModelError = np.array(ModelError)

#Runs fullmodel into future
fullmodel.time = 1000
S, I, D, H, E = fullmodel.simplesimulation()[0:5]

print('__________________________________________________________________________________________________')
print(f'Initial Error: {ModelError[0]:.20f} | Final Error: {ModelError[t-1]}')
print(
    f'Parameters: alpha={fullmodel.alpha},'
    f'beta={fullmodel.beta},' 
    f'gamma={fullmodel.gamma},' 
    f'delta={fullmodel.delta},' 
    f'zeta={fullmodel.zeta},' 
    f'omega={fullmodel.omega}')

print(
    f'Final statistics: S={S[fullmodel.time - 1]},'
    f'I={I[fullmodel.time - 1]},' 
    f'D={D[fullmodel.time - 1]},' 
    f'H={H[fullmodel.time - 1]},' 
    f'E={E[fullmodel.time - 1]}')

#Writes parameters to file
with open('parameterssimple.txt', 'w') as savefile:
    savefile.write(
        f'{fullmodel.alpha},'
        f'{fullmodel.beta},'
        f'{fullmodel.gamma},'
        f'{fullmodel.delta},'
        f'{fullmodel.zeta},'
        f'{fullmodel.omega}')

plt.figure(1)
plt.plot(ModelError)

plt.figure(2)
plt.plot(Dtrue)
plt.plot(D)

plt.figure(3)
plt.plot(Etrue)
plt.plot(E)

plt.figure(4)
plt.plot(Htrue)
plt.plot(H)

plt.figure(5)
plt.plot(S, label = 'Susceptible')
plt.plot(I, label = 'Infected')
plt.plot(D, label = 'Diagnosed')
plt.plot(H, label = 'Healed')
plt.plot(E, label = 'Expired')
plt.legend()

plt.show()
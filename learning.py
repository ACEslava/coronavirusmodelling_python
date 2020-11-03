import matplotlib.pyplot as plt
import numpy as np
from SIDHE import SIDHE

#Import real datasets
Dtrue = np.genfromtxt('COVID19-US.csv', delimiter=',')
Etrue =  np.genfromtxt('COVID19-US_Deaths.csv', delimiter=',')
Htrue = np.genfromtxt('COVID19-US_Recovered.csv', delimiter=',')
time = len(Dtrue)

#Initial parameter guess
model = SIDHE(0.003, 0.005, 0.04, 0.0001, 0.004, 0.0001, time)

ModelError = []
t = 1
time = len(Dtrue)

#Initial Error
S,I,D,H,E,size = model.fullsimulation()
Dtrue = np.divide(Dtrue, model.N)
Htrue = np.divide(Htrue, model.N)
Etrue = np.divide(Etrue, model.N)

ModelError.append(model.error(Dtrue, D) + model.error(Htrue, H) + model.error(Etrue, E)) 
ModelError.append(0)
ErrorDelta = []
ErrorDelta.append(0)
ErrorCheck = 0

while ErrorCheck < 2:

    #Tests error of each iteration
    S,I,D,H,E,size = model.fullsimulation()

    ModelError.append(model.error(Dtrue, D) + model.error(Htrue, H) + model.error(Etrue, E))

    
    #Changes parameters by small amount
    ChangeD = [0]*6
    ChangeH = [0]*6
    ChangeE = [0]*6
    
    for element, letter in enumerate('abgdzo'):
        ChangeD[element], ChangeH[element], ChangeE[element] = model.fullsimulation(letter)[2:5]

    #Finds derivative of the error function
    DerModelA = model.derivative(Dtrue, D, ChangeD[0]) + model.derivative(Htrue, H, ChangeH[0]) + model.derivative(Etrue, E, ChangeE[0]) 
    DerModelB = model.derivative(Dtrue, D, ChangeD[1]) + model.derivative(Htrue, H, ChangeH[1]) + model.derivative(Etrue, E, ChangeE[1]) 
    DerModelG = model.derivative(Dtrue, D, ChangeD[2]) + model.derivative(Htrue, H, ChangeH[2]) + model.derivative(Etrue, E, ChangeE[2]) 
    DerModelD = model.derivative(Dtrue, D, ChangeD[3]) + model.derivative(Htrue, H, ChangeH[3]) + model.derivative(Etrue, E, ChangeE[3]) 
    DerModelZ = model.derivative(Dtrue, D, ChangeD[4]) + model.derivative(Htrue, H, ChangeH[4]) + model.derivative(Etrue, E, ChangeE[4]) 
    DerModelO = model.derivative(Dtrue, D, ChangeD[5]) + model.derivative(Htrue, H, ChangeH[5]) + model.derivative(Etrue, E, ChangeE[5]) 

    #Gradient Descent
    h = 1E-9
    model.alpha -= h*DerModelA
    model.beta -= h*DerModelB
    model.gamma -= h*DerModelG
    model.delta -= h*DerModelD
    model.zeta -= h*DerModelZ
    model.omega -= h*DerModelO
    
    ErrorDelta.append(ModelError[t] - ModelError[t-1])
    print(f'Iteration {t:4d} | Error: {ModelError[t]:.20f} | Delta: {ModelError[t] - ModelError[t-1]:.20f} | ErrorDelta: {ErrorDelta[t] - ErrorDelta[t-1]}')
    
    #Checks if overall error is decreasing or increasing
    if ErrorDelta[t] - ErrorDelta[t-1] >=0:
        ErrorCheck += 1
    else: ErrorCheck -= 1

    t+=1

ModelError = np.array(ModelError)

#Runs model into future
model.time = 1000
S,I,D,H,E,size = model.fullsimulation()

print('__________________________________________________________________________________________________')
print(f'Initial Error: {ModelError[0]:.20f} | Final Error: {ModelError[t-1]}')
print(f'Parameters: alpha={model.alpha}, beta={model.beta}, gamma={model.gamma}, delta={model.delta}, zeta={model.zeta}, omega={model.omega}')
print(f'Final statistics: S={S[model.time - 1]}, I={I[model.time - 1]}, D={D[model.time - 1]}, H={H[model.time - 1]}, E={E[model.time - 1]}')

#Writes parameters to file
with open('parameters.txt', 'w') as savefile:
    savefile.write(f'{model.alpha},{model.beta},{model.gamma},{model.delta},{model.zeta},{model.omega}')

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
import numpy as np
import matplotlib.pyplot as plt

#Import real datasets
Dtrue = np.genfromtxt('COVID19-US.csv', delimiter=',')
Etrue =  np.genfromtxt('COVID19-US_Deaths.csv', delimiter=',')
Htrue = np.genfromtxt('COVID19-US_Recovered.csv', delimiter=',')

class GrowingList(list):
    def __setitem__(self, index, value):
        if index >= len(self):
            self.extend([None]*(index + 1 - len(self)))
        list.__setitem__(self, index, value)

class SIDHE:
    time = len(Dtrue)
    N = 330508267
    seed = 1
    Delta = 0.00001

    def __init__(self, alpha, beta, gamma, delta, zeta, omega):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.zeta = zeta
        self.omega = omega

    def simulation(self, change=0):
        S = np.array(np.zeros(self.time), dtype = np.float64)
        I = np.array(np.zeros(self.time), dtype = np.float64)
        D = np.array(np.zeros(self.time), dtype = np.float64)
        H = np.array(np.zeros(self.time), dtype = np.float64)
        E = np.array(np.zeros(self.time), dtype = np.float64)
        
        S[0] = self.N - self.seed
        I[0] = SIDHE.seed

        if change == 'a': self.alpha += self.Delta
        elif change == 'b': self.beta += self.Delta
        elif change == 'g': self.gamma += self.Delta
        elif change == 'd': self.delta += self.Delta
        elif change == 'z': self.zeta += self.Delta
        elif change == 'o': self.omega += self.Delta

        for i in range(time-1):
            S[i+1] = S[i] - S[i]*((self.beta/self.N)*I[i] + (self.alpha/self.N)*D[i])
            I[i+1] = I[i] + S[i]*((self.beta/self.N)*I[i] + (self.alpha/self.N)*D[i]) - self.gamma*I[i] - self.delta*I[i]
            D[i+1] = D[i] + self.gamma*I[i] - self.zeta*D[i] - self.omega*I[i]
            H[i+1] = H[i] + self.delta*I[i] + self.zeta*D[i]
            E[i+1] = E[i] + self.omega*D[i]

        return S,I,D,H,E
    
    def error(self, TrueVal, ApproximateVal):
        return np.sum(np.abs(TrueVal - ApproximateVal)/self.time)

    def derivative(self, TrueDataset, OriginalDataset, DeltaDataset):
        return((model.error(TrueDataset, DeltaDataset) - model.error(TrueDataset, OriginalDataset))/self.Delta)


#Initial parameter guess
model = SIDHE(0.055, 0.05, 0.04, 0.00001, 0.00001, 0.00001)

ModelError = GrowingList()
t = 1
time = len(Dtrue)
StopIterator = True

#Initial Error
S,I,D,H,E = model.simulation()
ModelError[0] = model.error(Dtrue, D) + model.error(Htrue, H) + model.error(Etrue, E) 
ModelError[1] = 0

while StopIterator == True:

    #Tests error of each iteration
    S,I,D,H,E = model.simulation()
    ModelError[t] = model.error(Dtrue, D) + model.error(Htrue, H) + model.error(Etrue, E)

    #Changes parameters by small amount
    Delta = 0.00001
    Sdummy, Idummy, Da, Ha, Ea = model.simulation('a')
    Sdummy, Idummy, Db, Hb, Eb = model.simulation('b')
    Sdummy, Idummy, Dg, Hg, Eg = model.simulation('g')
    Sdummy, Idummy, Dd, Hd, Ed = model.simulation('d')
    Sdummy, Idummy, Dz, Hz, Ez = model.simulation('z')
    Sdummy, Idummy, Do, Ho, Eo = model.simulation('o')

    #Finds derivative of the error function
    DerModelA = model.derivative(Dtrue, D, Da) + model.derivative(Htrue, H, Ha) + model.derivative(Etrue, E, Ea) 
    DerModelB = model.derivative(Dtrue, D, Db) + model.derivative(Htrue, H, Hb) + model.derivative(Etrue, E, Eb) 
    DerModelG = model.derivative(Dtrue, D, Dg) + model.derivative(Htrue, H, Hg) + model.derivative(Etrue, E, Eg) 
    DerModelD = model.derivative(Dtrue, D, Dd) + model.derivative(Htrue, H, Hd) + model.derivative(Etrue, E, Ed) 
    DerModelZ = model.derivative(Dtrue, D, Dz) + model.derivative(Htrue, H, Hz) + model.derivative(Etrue, E, Ez) 
    DerModelO = model.derivative(Dtrue, D, Do) + model.derivative(Htrue, H, Ho) + model.derivative(Etrue, E, Eo) 

    #Gradient Descent
    h = 1E-12
    model.alpha -= h*DerModelA
    model.beta -= h*DerModelB
    model.gamma -= h*DerModelG
    model.delta -= h*DerModelD
    model.zeta -= h*DerModelZ
    model.omega -= h*DerModelO

    print(f'Iteration {t:4d} | Error: {ModelError[t]:.20f} | Delta: {ModelError[t] - ModelError[t-1]:.20f}')

    #Checks if overall error is decreasing or increasing
    if ModelError[t-1] >= ModelError[t]:
        StopIterator = True
    else: StopIterator = False
    t+=1

ModelError = np.array(ModelError)
    
S,I,D,H,E = model.simulation()

print('__________________________________________________________________________________________________')
print(f'Initial Error: {ModelError[0]:.20f} | Final Error: {ModelError[t-1]}')
print(f'Parameters: alpha={model.alpha}, beta={model.beta}, gamma={model.gamma}, delta={model.delta}, zeta={model.zeta}, omega={model.omega}')
print(f'Final statistics: S={S[model.time - 1]}, I={I[model.time - 1]}, D={D[model.time - 1]}, H={H[model.time - 1]}, E={E[model.time - 1]}')


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
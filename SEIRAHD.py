import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
import scipy as sci
import scipy.spatial as sp
import random

alpha = 0.1
beta = 0.1
gamma = 0.08
delta = 0.5
theta = 0.4
Seeds = 6
TestingError = 0.7
Asiz = 5000
GroceryStore = 0.01

#Creates initial graph
coords = coordsave = np.random.uniform(-1,1,(Asiz,2))
proximity = 0.1
Distance = sp.distance.cdist(coords, coords, metric='euclidean')
A = Distance < proximity

#Initialising variables
E = np.zeros(Asiz)
for seed in range(1,Seeds+1):
    E[int(np.ceil((Asiz-1)*random.random()))] = 1

S = np.ones(Asiz) - E
I = As = R = H = D = NewS = NewE = NewI = NewA = NewR = NewH = NewD = np.zeros(Asiz)
SumS = SumE = SumI = SumR = SumA = SumH = SumD = np.zeros(1000)
t = 0
ACurrent = A
mask = lockdown = np.zeros(400)
lockdowndays = zerodays = FullLockdown = RestrictedLockdown = NoLockdown = 0
HRandom = np.random.random_sample(Asiz)

while np.sum(E) + np.sum(As) + np.sum(I) > 0:
    
    #Random Movement
    coords = coordsave = coords + np.random.uniform(-0.1,0.1,(Asiz,2))
    coords[coords>1] = -1
    coords[coords<1] = 1

    #Grocery Store
    for i in range(0, Asiz):
        if random.random() < GroceryStore:
            coords[i,0] = random.uniform(-0.0001,0.0001)
            coords[i,1] = random.uniform(-0.0001,0.0001)
    
    #Creates adjacency matrix for nodes
    Distance = sp.distance.cdist(coords, coords, metric='euclidean')
    A = Distance < proximity

    #Mask wearers
    Masks = np.zeros(Asiz)
    for maskcount in range(1,int(np.floor(Asiz*0.5))+1):
        Masks[int(np.ceil((Asiz-1)*random.random()))] = 1
    
    #Population Counters
    SumS[t] = np.sum(S)
    SumE[t] = np.sum(E)
    SumA[t] = np.sum(As)
    SumI[t] = np.sum(I)
    SumH[t] = np.sum(H)
    SumR[t] = np.sum(R)
    SumD[t] = np.sum(D)

    #Counts for days w/o infected
    if SumI[t] == 0:
        zerodays += 1
    else:
        zerodays = 0

    #Lockdown strategies
    if (TestingError * SumI[t]/Asiz) > 0.0025: #Full Lockdown
        proximity = 0.06
        movement = 0.4
        lockdown[t] = 1
        GroceryStore = 0.005
    elif zerodays > 14: #No Lockdown
        proximity = 0.1
        movement = 1
        lockdown[t] = 0
        locdkwndays = 0
        GroceryStore = 0.01
    elif t == 1:
        lockdown[1] = 0
    else:
        lockdown[t] = lockdown[t-1]
    
    if lockdowndays > 60: #Restricted Lockdown
        proximity = 0.08
        movement = 0.7
        lockdown[t] = 0.5
        lockdowndays = 0
        GroceryStore = 0.08
    
    #Updates Lockdown counters
    if lockdown[t] == 1:
        FullLockdown += 1
        lockdowndays += 1
    elif lockdown[t] == 0.5:
        RestrictedLockdown += 1
    else:
        NoLockdown += 1

    #Mask Strategy
    if (TestingError+SumI[t])/Asiz > 0.00000094285:
        mask[t] = 1
    elif zerodays > 18:
        mask[t] = 0
    elif t==1:
        mask[1] = 0
    else:
        mask[t] = mask[t-1]

    #Quarantine
    ToBeQuarantined = np.array([i for (i,j) in enumerate(I) if j == 1]).astype(int)
    A[ToBeQuarantined] = A[:, ToBeQuarantined] = 0

    #Infected Neighbors
    INeighbors = np.logical_or(I,As)
    INeighbors_Mask = np.dot(A*INeighbors, Masks)
    INeighbors_NMask = INeighbors - INeighbors_Mask

    #Who will be infected
    if mask[t] == 1:
        NewE = np.random.random_sample(Asiz) < 1 - np.dot(np.power((1-(0.7*Masks)*beta), INeighbors_Mask), np.power((1-(0.05-0.3*Masks)*beta), INeighbors_Mask))
    else:
        NewE = np.random.random_sample(Asiz) < 1 - 1 - np.power(1-beta, INeighbors)

    #S to E
    NewE = NewE.astype(bool) & S.astype(bool)

    #E to I or A
    ERandom = np.random.random_sample(Asiz)
    NewI = (ERandom < alpha).astype(bool) & E.astype(bool)
    NewA = (ERandom > 1-gamma).astype(bool) & E.astype(bool)

    #I to H
    IRandom = np.random.random_sample(Asiz) 
    age85 = I.astype(bool) & (HRandom < 0.02).astype(bool) & (IRandom < 0.5).astype(bool)
    age75 = I.astype(bool) & (HRandom > 0.02).astype(bool) & (HRandom < 0.06).astype(bool) & (IRandom < 0.43).astype(bool)
    age65 = I.astype(bool) & (HRandom > 0.06).astype(bool) & (HRandom < 0.16).astype(bool) & (IRandom < 0.3).astype(bool)
    ageother = I.astype(bool) & (HRandom > 0.16).astype(bool) & (IRandom < 0.10).astype(bool)

    NewH = age85 | age75 | age65 | ageother

    #I or H or A to R

    ItoR = (IRandom > 1-delta).astype(bool) & I.astype(bool)
    HtoR = (HRandom < theta).astype(bool) & H.astype(bool)
    AtoR = (np.random.random_sample(Asiz) > delta).astype(bool) & As.astype(bool)
    NewR = ItoR | HtoR | AtoR

    #H to D
    anotherrand = np.random.random_sample(Asiz) 
    age85 = I.astype(bool) & (HRandom < 0.02).astype(bool) & (anotherrand < 0.12).astype(bool)
    age75 = I.astype(bool) & (HRandom > 0.02).astype(bool) & (HRandom < 0.06).astype(bool) & (anotherrand < 0.06).astype(bool)
    age65 = I.astype(bool) & (HRandom > 0.06).astype(bool) & (HRandom < 0.16).astype(bool) & (anotherrand < 0.028).astype(bool)
    ageother = I.astype(bool) & (HRandom > 0.16).astype(bool) & (anotherrand < 0.002).astype(bool)

    NewD = ~NewR & (age85 | age75 | age65 | ageother)

    #Update Indicators
    S = S - NewE.astype(int)
    E = E + NewE.astype(int) - (NewI.astype(int) + NewA.astype(int))
    I = I + NewI.astype(int) - ItoR.astype(int) - NewH.astype(int)
    As = As + NewA.astype(int) - AtoR.astype(int)
    H = H + NewH.astype(int) - (HtoR.astype(int) + NewD.astype(int))
    R = R + NewR.astype(int)
    D = D + NewD.astype(int)

    coords = coordsave
    t += 1
    print(t)

print("Total Deaths:", SumD[t])
print("Days with COVID:", t)
plt.spy(A)
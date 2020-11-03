import numpy as np
import scipy
import matplotlib.pyplot as plt
import networkx as nx
import random
import math

#Import real datasets
Dtrue = np.genfromtxt('COVID19-US.csv', delimiter=',')
Etrue =  np.genfromtxt('COVID19-US_Deaths.csv', delimiter=',')
Htrue = np.genfromtxt('COVID19-US_Recovered.csv', delimiter=',')
time = len(Dtrue)

class GrowingList(list):
    def __setitem__(self, index, value):
        if index >= len(self):
            self.extend([None]*(index + 1 - len(self)))
        list.__setitem__(self, index, value)

class SIDHE:
    N = 330508267
    seed = 1
    Delta = 0.00001

    def __init__(self, alpha, beta, gamma, delta, zeta, omega, time):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.zeta = zeta
        self.omega = omega
        self.time = time
    
    def buckettobucket(self, bucket_from, probability, size):
        return 
    def fullsimulation(self, change=0):

        # S = np.array(np.zeros(self.time), dtype = np.float64)
        # I = np.array(np.zeros(self.time), dtype = np.float64)
        # D = np.array(np.zeros(self.time), dtype = np.float64)
        # H = np.array(np.zeros(self.time), dtype = np.float64)
        # E = np.array(np.zeros(self.time), dtype = np.float64)
        
        # S[0] = self.N - self.seed
        # I[0] = SIDHE.seed

        # if change == 'a': self.alpha += self.Delta
        # elif change == 'b': self.beta += self.Delta
        # elif change == 'g': self.gamma += self.Delta
        # elif change == 'd': self.delta += self.Delta
        # elif change == 'z': self.zeta += self.Delta
        # elif change == 'o': self.omega += self.Delta

        #Creates nodes w/ random coordinates on 2-D Cartesian plane btwn -1 and 1
        size = 100
        xcoord = np.random.uniform(-1.0, 1.0, size)
        ycoord = np.random.uniform(-1.0, 1.0, size)
        
        xcoord[xcoord>1] = -1
        xcoord[xcoord<-1] = 1
        ycoord[ycoord>1] = -1
        ycoord[ycoord<-1] = 1

        #Creates edges based on threshold distances btwn nodes
        proximity = 0.3
        AdjacencyMatrix = np.zeros((size, size))
        Distance = np.zeros((size, size))

        for j in range(1, size):
            for i in range(1, size):
                Distance[i,j] = math.sqrt((xcoord[i]-xcoord[j])**2 + (ycoord[i]-ycoord[j])**2)
                AdjacencyMatrix[i,j] = Distance[i,j] < proximity

        #Generates NetworkX graph
        Population = nx.convert_matrix.from_numpy_array(AdjacencyMatrix)

        #Initialises indicator vectors
        
        I = np.array(np.zeros(size), dtype=bool)
        for x in random.sample(range(size), self.seed):
            I[x] = 1

        S = np.array(~I, dtype=bool)
        D = np.array(np.zeros(size), dtype=bool)
        H = np.array(np.zeros(size), dtype=bool)
        E = np.array(np.zeros(size), dtype=bool)

    
        t=0

        SumS = []
        SumI = []
        SumD = []
        SumH = []
        SumE = []

        while(sum(I) + sum(D) > 0):

            SumS.append(sum(S))
            SumI.append(sum(I))
            SumD.append(sum(D))
            SumH.append(sum(H))
            SumE.append(sum(E))

            #S to I
            InfectedNeighbors = np.dot(AdjacencyMatrix, I.astype(int))
            InfectedDiagnosed = np.dot(AdjacencyMatrix, D.astype(int))
            NewI = np.less(np.random.rand(size), np.multiply((1-np.power((1-self.beta), InfectedNeighbors)), 1-np.power((1-self.alpha), InfectedDiagnosed)))
            NewI = np.logical_and(NewI, S.astype(bool))

            NewD = np.logical_and((np.random.rand(size) < self.gamma), I.astype(bool))
            DtoH = np.logical_and((np.random.rand(size) < self.zeta), D.astype(bool))
            ItoH = np.logical_and(np.logical_and((np.random.rand(size) < self.delta), I.astype(bool)), ~NewD)
            NewE = np.logical_and(np.logical_and((np.random.rand(size) < self.omega), D.astype(bool)), ~DtoH)

            S = S.astype(int) - NewI.astype(int)
            I = I.astype(int) + NewI.astype(int) - NewD.astype(int) - ItoH.astype(int)
            D = D.astype(int) + NewD.astype(int) - DtoH.astype(int)
            H = H.astype(int) + ItoH.astype(int) + DtoH.astype(int)
            E = E.astype(int) + NewE.astype(int)

            t += 1
        
        return SumS

    def error(self, TrueVal, ApproximateVal):
        return np.sum(np.abs(TrueVal - ApproximateVal)/self.time)

    def derivative(self, TrueDataset, OriginalDataset, DeltaDataset):
        return((self.error(TrueDataset, DeltaDataset) - self.error(TrueDataset, OriginalDataset))/self.Delta)


model = SIDHE(0.055, 0.005, 0.04, 0.00001, 0.00001, 0.00001, time)

print(model.fullsimulation())
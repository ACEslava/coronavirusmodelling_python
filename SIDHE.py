import numpy as np
import networkx as nx
import random
import math

class SIDHE:
    N = 330508267
    seed = 1
    Delta = 0.00001
    SimGraph = nx.generators.random_graphs.fast_gnp_random_graph(500, 0.01)
    AdjacencyMatrix = nx.convert_matrix.to_numpy_array(SimGraph)
    size = nx.classes.function.number_of_nodes(SimGraph)

    def __init__(self, alpha, beta, gamma, delta, zeta, omega, time):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.zeta = zeta
        self.omega = omega
        self.time = time
    
    def fullsimulation(self, change=0):
        #Creates nodes w/ random coordinates on 2-D Cartesian plane btwn -1 and 1
        # size = 1000
        # xcoord = np.random.uniform(-1.0, 1.0, size)
        # ycoord = np.random.uniform(-1.0, 1.0, size)
        
        # xcoord[xcoord>1] = -1
        # xcoord[xcoord<-1] = 1
        # ycoord[ycoord>1] = -1
        # ycoord[ycoord<-1] = 1

        # #Creates edges based on threshold distances btwn nodes
        # proximity = 0.1
        # AdjacencyMatrix = np.zeros((size, size))
        # Distance = np.zeros((size, size))

        # for j in range(1, size):
        #     for i in range(1, size):
        #         Distance[i,j] = math.sqrt((xcoord[i]-xcoord[j])**2 + (ycoord[i]-ycoord[j])**2)
        #         AdjacencyMatrix[i,j] = Distance[i,j] < proximity

        # #Generates NetworkX graph
        # Population = nx.convert_matrix.from_numpy_array(AdjacencyMatrix)

        #Checks if values need to be changed
        if change == 'a': self.alpha += self.Delta
        elif change == 'b': self.beta += self.Delta
        elif change == 'g': self.gamma += self.Delta
        elif change == 'd': self.delta += self.Delta
        elif change == 'z': self.zeta += self.Delta
        elif change == 'o': self.omega += self.Delta

        #Initialises indicator vectors
        I = np.array(np.zeros(self.size), dtype=int)
        for x in random.sample(range(self.size), self.seed):
            I[x] = 1

        S = np.array(~I.astype(bool), dtype=int)
        D = np.array(np.zeros(self.size), dtype=int)
        H = np.array(np.zeros(self.size), dtype=int)
        E = np.array(np.zeros(self.size), dtype=int)

    
        t=0

        SumS = []
        SumI = []
        SumD = []
        SumH = []
        SumE = []

        while t < self.time:

            SumS.append(sum(S))
            SumI.append(sum(I))
            SumD.append(sum(D))
            SumH.append(sum(H))
            SumE.append(sum(E))

            
            #S to I
            InfectedNeighbors = np.dot(self.AdjacencyMatrix, I)
            DiagnosedNeighbors = np.dot(self.AdjacencyMatrix, D)
            NewI = np.random.rand(self.size) < ((1 - np.power((1-self.beta), InfectedNeighbors)) + (1 - np.power((1-self.alpha), DiagnosedNeighbors)))
            NewI = np.logical_and(NewI, S.astype(bool))

            #I to H or D
            RandomNumberI = np.random.rand(self.size)
            ItoH = np.logical_and((RandomNumberI < self.delta), I.astype(bool))
            NewD = np.logical_and((RandomNumberI > 1 - self.gamma), I.astype(bool))

            #D to H or E
            RandomNumberD = np.random.rand(self.size)
            DtoH = np.logical_and((RandomNumberD < self.zeta), D.astype(bool))
            NewE = np.logical_and((RandomNumberD > 1 - self.omega), D.astype(bool))

            #Update Indicator Vectors
            S = S - NewI
            I = I + NewI - ItoH - NewD
            D = D + NewD - DtoH - NewE
            H = H + ItoH + DtoH
            E = E + NewE

            t += 1
        
        #Makes sums a proportion of population
        SumS = np.divide(SumS, self.size)
        SumI = np.divide(SumI, self.size)
        SumD = np.divide(SumD, self.size)
        SumH = np.divide(SumH, self.size)
        SumE = np.divide(SumE, self.size)

        return SumS, SumI, SumD, SumH, SumE, self.size

    def simplesimulation(self, change=0):
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

        for i in range(self.time-1):
            S[i+1] = S[i] - S[i]*((self.beta/self.N)*I[i] + (self.alpha/self.N)*D[i])
            I[i+1] = I[i] + S[i]*((self.beta/self.N)*I[i] + (self.alpha/self.N)*D[i]) - self.gamma*I[i] - self.delta*I[i]
            D[i+1] = D[i] + self.gamma*I[i] - self.zeta*D[i] - self.omega*I[i]
            H[i+1] = H[i] + self.delta*I[i] + self.zeta*D[i]
            E[i+1] = E[i] + self.omega*D[i]

        return S,I,D,H,E
    
    def error(self, TrueVal, ApproximateVal):
        return np.sum(np.abs(TrueVal - ApproximateVal)/self.time)

    def derivative(self, TrueDataset, OriginalDataset, DeltaDataset):
        return((self.error(TrueDataset, DeltaDataset) - self.error(TrueDataset, OriginalDataset))/self.Delta)

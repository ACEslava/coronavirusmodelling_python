import numpy as np
import matplotlib.pyplot as plt

beta = 0.25
omega = 0.15
N = 330508267
time = 500
seed = 1
S = np.array(np.zeros(time), dtype = np.float64)
I = np.array(np.zeros(time), dtype = np.float64)
R = np.array(np.zeros(time), dtype = np.float64)

S[0] = N - seed
I[0] = seed

for i in range(time-1):
    S[i+1] = S[i] - S[i]*(1-((1-beta/N)**I[i]))
    I[i+1] = I[i] + S[i]*(1-((1-beta/N)**I[i])) - omega*I[i]
    R[i+1] = R[i] + omega*I[i]

S = [i / N for i in S]
I = [i / N for i in I]
R = [i / N for i in R]

plt.figure(0)
plt.plot(S, label = 'Susceptible', color='m')
plt.plot(I, label = 'Infected', color='g')
plt.plot(R, label = 'Recovered', color='k')
plt.legend()
plt.title("Time Evolution of the SIR Model")
plt.xlabel("Time")
plt.ylabel("Proportion of Population")

plt.show()
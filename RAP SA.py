import math as m
import random as rand

# Reliability of each component in series
r = [.7,.75,.8,.85,.9,.95]

# Cost of each component in series (strong positive corelation)
c = [1,2,3,4,5,6]

# Cost of each component in series (strong negative corelation)
#c = [6,5,4,3,2,1]

# Constants
MAX_TEMP = 100
BOLTZMAN = .01

# Cost bound
max_cost = 250

# Approximate target for convergence calculations
L = 0.9999

# Normalize a given vector
def normalizeVector(v):
    l = m.sqrt(sum([x**2 for x in v]))
    v = [x/l for x in v]
    return v

def normalizeAndScale(v):
    v_n = normalizeVector(v)
    dot_p = sum([v_n[i] * c[i] for i in range(len(c))])
    if dot_p == 0:
        return None
    k = max_cost / dot_p
    v = [ m.floor(k * v_n[i]) for i in range(len(c))]
    return v

# Calculate reliability of the entire system
def getReliability(r,count):
    reliability = 1
    v = [0 for x in range(len(r))]
    for i in range(len(r)):
        x = 1-((1-r[i]) ** (count[i]+1))
        reliability *= x
        v[i] = x
    return reliability,v

def SA(lastcount,lastreliability,t):
    # Fill the count vector with random numbers
    count = [rand.random() for x in range(len(r))]
	
    # Normalize the vector and scale it to max cost
    count = normalizeAndScale(count)

    # Apply the simulated annealing formula
    reliability,_ = getReliability(r,count)
    if reliability >= lastreliability:
        return count
    else:
        p = m.exp((reliability-lastreliability)/(BOLTZMAN*t))        
        if rand.random() < p:
            return count
        else:
            return lastcount
                
ev = []
def cycle(r,c):
    # Initialize the number of redundant components to 0
    count = [0 for x in range(len(r))]    

    rv = []
    cv = []
    
    for t in range(MAX_TEMP):
        reliability,_ = getReliability(r,count)
		
	# Calculate cost of the system
        cost = 0
        for i in range(len(r)):
            cost += c[i] * count[i]
        print(reliability,cost,count)
        
        rv.append(reliability)
        cv.append(cost)
        ev.append(abs(reliability-L))
        # Get count from simulated annealing algorithm
        count = SA(count,reliability,MAX_TEMP-t)
                
    return rv,cv

rv,cv = cycle(r,c)

import matplotlib.pyplot as plt
import numpy as np
# Reliability
'''
plt.plot(range(len(rv)),rv)
plt.xlabel("Iterations")
plt.ylabel("Reliability")
plt.title("Reliability")
'''
#Cost
'''
plt.plot(range(len(cv)),cv)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("SA cost")
'''

# Convergence

u = [ev[i]/(ev[i-1]) for i in range(1,len(ev))]
A = np.vstack([range(len(u)), np.ones(len(u))]).T
m, c = np.linalg.lstsq(A, u)[0]
x = range(len(u))
plt.scatter(x,u)
plt.plot(x, m * x + c,label="Fitted line")
plt.xlabel("Iterations")
plt.ylabel("e[i]/e[i-1]")
plt.title("Convergence rate")

plt.show()

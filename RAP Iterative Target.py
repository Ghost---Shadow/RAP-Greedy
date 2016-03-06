import math as m

# Reliability of each component in series
r = [.7,.75,.8,.85,.9,.95]

# Cost of each component in series (strong positive corelation)
#c = [1,2,3,4,5,6]

# Cost of each component in series (strong negative corelation)
c = [6,5,4,3,2,1]

# Target reliability and tolerance
target = m.exp(m.log(0.9999999)/len(r))

# Calculate reliability of the entire system
def getReliability(r,count):
    reliability = 1
    v = [0 for x in range(len(r))]
    for i in range(len(r)):
        x = 1-((1-r[i]) ** (count[i]+1))
        reliability *= x
        v[i] = x
    return reliability,v

ev = []

def cycle(r,c,target):
    # Initialize the number of redundant components to 0
    count = [0 for x in range(len(r))]
    last_reliability = 0

    rv = []
    cv = []
    
    for i in range(len(r)):
        # Match the reliability of this component with the target reliability
        count[i] = m.floor(m.log(1-target)/m.log(1-r[i]))

        # Calculate reliability for plotting graphs        
        reliability,_ = getReliability(r,count)
		
	# Calculate cost of the system for graphs
        cost = 0
        for i in range(len(r)):
            cost += c[i] * count[i]

        # Vectors for plotting graphs
        print(reliability,cost,count)       
        rv.append(reliability)
        cv.append(cost)
        ev.append(abs(reliability-target))
    return rv,cv,cost

rv,cv,cost = cycle(r,c,target)

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

plt.plot(range(len(cv)),cv)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Target cost")


# Convergence
'''
u = [ev[i]/(ev[i-1]) for i in range(1,len(ev))]
A = np.vstack([range(len(u)), np.ones(len(u))]).T
m, c = np.linalg.lstsq(A, u)[0]
x = range(len(u))
plt.scatter(x,u)
plt.plot(x, m * x + c,label="Fitted line")
plt.xlabel("Iterations")
plt.ylabel("e[i]/e[i-1]")
plt.title("Convergence rate")
'''
plt.show()

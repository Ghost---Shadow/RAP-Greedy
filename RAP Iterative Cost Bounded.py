import math as m

MAX_INT = 99999

# Reliability of each component in series
r = [.7,.75,.8,.85,.9,.95]

# Cost of each component in series (strong positive corelation)
#c = [1,2,3,4,5,6]

# Cost of each component in series (strong negative corelation)
c = [6,5,4,3,2,1]

# Total resource that can be consumed
MAX_R = resource = 250

# Precalculated target for convergence calculations
L = 0.9999999803262839  

# Calculate the reliability of the system
def getReliability(r,count):
    reliability = 1
    v = [0 for x in range(len(r))]
    for i in range(len(r)):
        x = 1-((1-r[i]) ** (count[i]+1))
        reliability *= x
        v[i] = x
    return reliability,v

# Find the component with lowest reliability that can be afforded
def lowest(v,c,resource):
    m = MAX_INT
    index = -1
    for i in range(len(v)):
        if v[i] < m and c[i] <= resource:
            index = i
            m = v[i]
    return index
ev = []
def cycle(r,c,resource):
    # Number of redundant components
    count = [0 for x in range(len(r))]

    rv = []
    drv = []
    cv = []    

    lastreliability = 0
    
    for _ in range(resource):
        reliability,v = getReliability(r,count)

        l_index = lowest(v,c,resource)
        if l_index == -1:
            break
        
	# Increment count of that component by 1
        count[l_index] += 1
		
	# Update available resources
        resource -= c[l_index]
        
	# Calculate total cost
        cost = 0
        for i in range(len(r)):
            cost += c[i] * count[i]

        # Analysis and graph plotting vectors
        print(reliability,resource, count)
        cv.append(MAX_R - resource)
        rv.append(reliability)
        drv.append(reliability-lastreliability)        
        ev.append(abs(reliability-L))
        lastreliability = reliability
        
    return rv,drv,cv

rv,drv,cv = cycle(r,c,resource)


import matplotlib.pyplot as plt
import numpy as np
# Reliability
'''
plt.plot(range(len(rv)),rv)
plt.xlabel("Iterations")
plt.ylabel("Reliability")
plt.title("Cost Bounded Reliability")
'''

#Cost

plt.plot(range(len(cv)),cv)
plt.xlabel("Iterations")
plt.ylabel("Used resources")
plt.title("Cost Bounded cost")


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

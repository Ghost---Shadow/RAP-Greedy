import math as m
import random as rand

# Reliability of each component in series
r = [.7,.75,.8,.85,.9,.95]

# Cost of each component in series (strong positive corelation)
#c = [1,2,3,4,5,6]

# Cost of each component in series (strong negative corelation)
c = [6,5,4,3,2,1]

# Constants
POOL_SIZE = 10
MAX_GENERATIONS = 100
appx_target = .9999
MUTATION = .001

# Cost bound
max_cost = 250

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
    #print(dot_p,k)
    #print(v_n)
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

def initPool():
    pool = []
    for _ in range(POOL_SIZE):
        # Create a random gene vector
        count = [rand.random() for x in range(len(r))]

        # Normalize it and then scale it to max cost
        count = normalizeAndScale(count)

        # Add it to progeny
        pool.append(count)
    return pool

def crossover(p1,p2):
    child1 = [0 for _ in range(len(p1))]
    child2 = [0 for _ in range(len(p2))]

    factor = rand.random()
    
    # Interpolate
    for i in range(len(p1)):
        child1[i] = factor * p1[i] + (1-factor)*p2[i]
        child1[i] = factor * p2[i] + (1-factor)*p1[i]

    # Mutation Chance
    if rand.random() < MUTATION:
        i = rand.randint(0,len(p1)-1)
        child1[i] += rand.random() * max_cost
        i = rand.randint(0,len(p2)-1)
        child2[i] += rand.random() * max_cost
        #print("Mutation!")   
    
    # Error handling
    if sum(child1) == 0 or sum(child2) == 0:
        return p1,p2
    
    # Normalize and scale the child vectors
    child1 = normalizeAndScale(child1)
    child2 = normalizeAndScale(child2)    

    return child1,child2

def rouletteSelect(weight):
    value = rand.uniform(0,sum(weight))   
    for i in range(len(weight)):
        value -= weight[i]        
        if value <= 0.0:
            return i
        
    return len(weight) - 1

def getNextGen(lastGen):    
    nextGen = []
    fitness = [0 for _ in range(POOL_SIZE)]    
    
    # Calculate fitness
    for i in range(POOL_SIZE):
        fitness[i],_ = getReliability(r,lastGen[i])
        #print(lastGen[i],fitness[i])

    # Sort the pool by fitness
    pool = [(y,x) for (y,x) in sorted(zip(fitness,lastGen))]
    fitness = sorted(fitness)    
    
    # Append alpha
    nextGen.append(pool[-1][1])
    #print("Alpha: ",nextGen[0])

    while abs(sum(fitness)) > 0:
        #print(abs(sum(fitness)))
        # Roulette select the two parents
        p1 = rouletteSelect(fitness)
        fitness[p1] = 0 # Invalidate p1
        p2 = rouletteSelect(fitness)
        fitness[p2] = 0 # Invalidate p2
        
        # Get parent genomes
        parent1 = pool[p1][1]
        parent2 = pool[p2][1]       

        # Mate
        child1,child2 = crossover(parent1,parent2)
        nextGen.append(child1)
        nextGen.append(child2)

        #print(p1,p2)

    fitness.append(0)
    
    # Remove the least fit child
    for i in range(POOL_SIZE + 1):
        fitness[i],_ = getReliability(r,nextGen[i])
        #print(nextGen[i],fitness[i])
    least_fit_index = fitness.index(min(fitness))
    nextGen.remove(nextGen[least_fit_index])

    return nextGen
       
ev = []
rv = []
cv = []

def cycle(r,c):
    
    # Create progeny
    pool = initPool()    
    for _ in range(MAX_GENERATIONS):
        pool_reliability = []
        for v in pool:
            pr,_ = getReliability(r,v)
            pool_reliability.append(pr)       

        pool = getNextGen(pool)
        
        # Graphing and analysis
        rv.append(sum(pool_reliability)/len(pool_reliability))
        best_index = pool_reliability.index(max(pool_reliability))
        best = pool[best_index]
        cost = 0
        for i in range(len(best)):
            cost += c[i] * best[i]
        cv.append(cost)
        ev.append(abs(rv[-1]-appx_target))

    alpha_reliability,_ = getReliability(r,pool[0])
    alpha_cost = sum([c[i] * pool[0][i] for i in range(len(c))])
    print(alpha_reliability, alpha_cost, pool[0])
    return rv,cv

rv,cv = cycle(r,c)

import matplotlib.pyplot as plt
import numpy as np

# Reliability
plt.plot(range(len(rv)),rv)
plt.xlabel("Iterations")
plt.ylabel("Reliability")
plt.title("Reliability")

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

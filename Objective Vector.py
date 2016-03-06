import math as m

# Reliability of each component
r = [.7, .75]

# Cost of each component
c = [2,3]

# Max cost
max_cost = 20

def getReliability(r,count):
    reliability = 1
    for i in range(len(r)):
        x = 1-((1-r[i]) ** (count[i]+1))
        reliability *= x
    return reliability

count_v = []
rv = []

# Parameterize the function
for x in range(0,max_cost+1):
    y = int(m.floor((max_cost - c[0] * x)/c[1]))
    if y < 0:
        continue
    count_v.append([x,y])
    rv.append(getReliability(r,[x,y]))

# Normalize and scale by reliability
gv = []
for i in range(len(rv)):
    x,y = count_v[i][0],count_v[i][1]
    l = (x**2 + y**2) ** .5
    nx,ny = rv[i]*x/l,rv[i]*y/l
    gv.append([nx,ny])

# Plot graphs
import numpy as np
import matplotlib.pyplot as plt
soa = np.array( [[0,0,x,y] for x,y in gv]) 
X,Y,U,V = zip(*soa)
fig = plt.figure()
ax = plt.gca()
ax.set_aspect('equal')
ax.quiver(X,Y,U,V,angles='xy',scale_units='xy',scale=1,facecolor='g')
for i in range(len(gv)):
    u,v = gv[i]
    x,y = count_v[i]
    rel = round(rv[i],3)
    plt.text(u, v, '<'+str(x)+','+str(y)+','+str(rel)+'>')
ax.set_xlim([0,1])
ax.set_ylim([0,1])
plt.grid(True)
circle=plt.Circle((0,0),1,color='r',fill=False)
ax.add_artist(circle)
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("Reliability")
plt.draw()
plt.show()
        
    
    
    
    

print("This program is for Homework 6: PDEs & Monte Carlo")
print("--------------------------------------------------")
print("I chose Question 10.3: Brownian Motion in 2-D.")
print("--------------------------------------------------")

import numpy
import matplotlib.pyplot
import vpython #import visual before random to avoid problems
import random

def walk(position,L,currentcolor):
    i = position.x
    j = position.y
    edgeflag=False
    # generate appropriate random direction
    if (i==0 or i==L or j==0 or j==L):
        edgeflag = True
        edgecolor = vpython.color.green
        if i==0: 
            r = random.choice(["north","east","south"])
        elif i==L: 
            r = random.choice(["north","south","west"])
        elif j==0:
            r = random.choice(["north","east","west"])
        else:
            r = random.choice(["east","south","west"])
    else:
        r = random.choice(["north", "east", "south", "west"])
    #move in that chosen direction
    if r=="east": 
        currentcolor = vpython.color.orange
        i+=1
    elif r=="west": 
        currentcolor = vpython.color.yellow
        i-=1
    elif r=="north": 
        currentcolor = vpython.color.red
        j+=1
    if r=="south": # South
        currentcolor = vpython.color.blue
        j-=1  
#    if r==1: # East
#        if i==L: 
#            currentcolor = vpython.color.green
#        else:
#            currentcolor = vpython.color.orange
#            i+=1
#    if r==2: # West
#        if i==0:
#            currentcolor = vpython.color.green
#        else:
#            currentcolor = vpython.color.yellow
#            i-=1
#    if r==3: # North
#        if j==L:
#            currentcolor = vpython.color.green
#            #continue
#        else:
#            currentcolor = vpython.color.red
#            j+=1
#    if r==4: # South
#        if j==0:
#            currentcolor = vpython.color.green
#            #continue
#        else:
#            currentcolor = vpython.color.blue
#            j-=1
    if edgeflag==True:
        return [vpython.vector(i,j,0),edgecolor]
    else:
        return [vpython.vector(i,j,0),currentcolor]

L = 11 # length of side of lattice
i = j = 5 # starting position of particle -- center of lattice
d = vpython.display(background=vpython.color.white) # for the animations, later
#d.autoscale = False
#d.background=vpython.color.white
currentcolor = vpython.color.white
position = vpython.vector(i,j,0) # 3-D coordinates of starting position
s = vpython.sphere(pos=position,radius=1.5,color=currentcolor)
steps = 100
for t in numpy.arange(steps):
    vpython.rate(10) # rate for visualization
    l = vpython.sphere(pos=position,radius=1,color=currentcolor)
    r = random.randint(1,4)
    #update position vector
    walk_list = walk(position,L,currentcolor)
    position = walk_list[0]
    currentcolor = walk_list[1]
final = vpython.sphere(pos=position,radius=1.5,color=vpython.color.black)

print("end")

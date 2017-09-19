print("This is a program to display the Mandlebrot Set!")
import numpy
from numba import jit # compile to make run faster
import matplotlib.pyplot # To plot the set
import matplotlib.image

# Definitions
# 1. Calculate the Mandlebrot Set
@jit
def mandlebrot(x_min, x_max, x_steps, y_min, y_max, y_steps, iterations): # iterations is the maximum number of iterations of adding the complex constant that we allow the computer to check if seeded complex number reaches 2 or not
    rl = numpy.linspace(x_min, x_max, x_steps, dtype=float)               #empty array for real values of constant
    im = numpy.linspace(y_min, y_max, y_steps, dtype=float)               #empty array for imgy values of constant
    n_iterations = numpy.zeros(shape=(x_steps, y_steps), dtype=int)       #empty array to take the no. of iterations
    for x_index in range(x_steps):                                        #do each step in x
        for y_index in range(y_steps):                                    #for every step in x, do each step in y
            z=0.+0.*1j                                                    #seeded complex number for the algorithm
            for n in range(iterations):                                   #start counting iterations...until we hit 2
                if (numpy.abs(z) >= 2):                                   #if our complex number is over or equal to 2 ...
                    n_iterations[x_index,y_index] = n                     # ... then record the number of iterations it took to get there
                    break                                                 # ... and stop looking at this (x, y) pair
                else:                                                     #if our z is less than 2 ... 
                    z = z*z + (rl[x_index] + im[y_index]*1j)              #... then keep going and add the constant
    return n_iterations                                                   #returns x_max-by-y_max, 2D array

# Logarithm for more intricate coloration
@jit
def get_logarithm_of_2_by_2_array(array, x_steps, y_steps):                    #takes a 2-by-2 array
    log_array = numpy.zeros(array.shape, dtype=float)                          #make an empty array to record the logarithms, same shape as input array
    for x_index in range(x_steps):                                             #do each step in x
        for y_index in range(y_steps):                                         #for every step in x, do each step in y
            if (array[x_index,y_index] <= 0):                                  #if the value in the original array is lessthan or equal to zero...
                log_array[x_index,y_index] = 0.                                #... then set the logarithm = 0.
            else:                                                              #if greater than zero...
                log_array[x_index,y_index] = numpy.log(array[x_index,y_index]) #... then do the logarithm
    return log_array                                                           #returns an array 


# Run the Program
if __name__ == "__main__":
# Let's Calculate the "whole" Mandlebrot set: that christmas-tree-like figure
    # x_min, x_max, x_steps = -2, 0.5, 5000
    # y_min, y_max, y_steps = -1.25, 1.25, 5000
    # iterations = 25
    
    # n_iterations = mandlebrot(x_min, x_max, x_steps, y_min, y_max, y_steps, iterations) #array of number of iterations it takes for z=0 to get to z=2
    # log_n_iterations = get_logarithm_of_2_by_2_array(n_iterations, x_steps, y_steps) #log of that array
    
# Zoom - in on a smaller window -- an edge of the figureset
    x_min2, x_max2, x_steps2 = 0.3, 0.4, 5000
    y_min2, y_max2, y_steps2 = 0.4, 0.5, 5000
    iterations2 = 25

    n_iterations_zoom = mandlebrot(x_min2, x_max2, x_steps2, y_min2, y_max2, y_steps2, iterations2)
    log_n_iterations_zoom = get_logarithm_of_2_by_2_array(n_iterations_zoom, x_steps2, y_steps2)

#Plotting
    # matplotlib.pyplot.imshow(log_n_iterations)
    # matplotlib.pyplot.hot()
    # matplotlib.pyplot.show()
    matplotlib.pyplot.imshow(log_n_iterations_zoom)
    matplotlib.pyplot.hot()
    matplotlib.pyplot.show()

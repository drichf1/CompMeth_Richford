# 6.16
print("This program is a tool to calculate the L-1 Lagrange Point\nfor a satellite between the Earth and the Moon.\n----------------")

# Import Modules
import time                      # for timing
start_time = time.time()
import numpy                     # for math analysis
# import matplotlib.pyplot       # for plotting and visualization
#import matplotlib.pylab         # for plotting and visualization
import_end_time = time.time()

# Definitions
def_start_time = time.time()
def physical_constants(): # makes an array of the relevant physical constants
    G = 6.674e-11 # m^3 kg^-1 s^-2
    M_earth = 5.974e24 # kg
    M_moon = 7.348e22 # kg
    R_earth_moon = 3.844e8 # m
    angular_veloc_moon = 2.6662e-6 # s^-1
    return [G, M_earth, M_moon, R_earth_moon, angular_veloc_moon]

def coefficients(x): #determines coefficients; takes 5-element array [ G, mEarth, mMooon, rEarth, omegaMoon]
    A = x[0]*x[1]
    B = x[0]*x[2]
    C = x[3]
    D = x[4]**2
    return [A,B,C,D]

def poly(coefficients,x): # graphs the polynomial; coefficients is a 4 element list: A,B,C,D
    '''
    Equation Given:
     G * M_earth        G * M_moon
    ------------- - ----------------- = omega^2 * r
        r^2          (R_earth - r)^2
    '''
    if ((hasattr(x,"__len__")) and ((not isinstance(x,str))==True)): #if x is a list, use this
        return coefficients[0]*(x[i]**(-2)) - coefficients[1]*((coefficients[2]-x[i])**(-2)) - x[i]*(coefficients[3]**2)
    else: #if x is a single value
        return coefficients[0]*(x**(-2)) - coefficients[1]*((coefficients[2]-x)**(-2)) - x*(coefficients[3]**2)

def derivpoly(coefficients,x): # x is a four-element list
    if ((hasattr(x,"__len__")) and ((not isinstance(x,str))==True)): #if x is a list, use this
        return (-2)*coefficients[0]*(x[i]**(-3)) - (-2)*coefficients[1]*((coefficients[2]-x[i])**(-3))*(-1) - coefficients[3]**2
    else: #if x is a single value
        return (-2)*coefficients[0]*(x**(-3)) - (-2)*coefficients[1]*((coefficients[2]-x)**(-3))*(-1) - coefficients[3]**2

def newton(coefficients,x): #newton's method
    f = poly(coefficients,x)
    fprime = derivpoly(coefficients,x)
    xprime = x-f/fprime
    return xprime

def_end_time = time.time()

# Main Program -- Calculate that L1 Point
main_start_time = time.time()
if __name__ == "__main__":
    constants = physical_constants()
    coefficients = coefficients(constants)
    rmin = 6371e3 # meters (radius of earth) #calculating the L1 point between the radii of the objects
    rmax = constants[4]-1737e3 #Earth-moon distance minus radius of moon
    N = 10000 #1000 is okay, 10000 is better, with no great loss of speed
    r = numpy.linspace(rmin,rmax,N) # array for our radii
    temp = numpy.zeros(r.shape,float) #array for our answers
#    p = numpy.zeros(r.shape,float) #array for polynomial
#    pprime = numpy.zeros(r.shape,float) #array for derivative
#    for i in range(N): #record our polynomials and derivative for checking
#        p[i] = poly(coefficients,r[i])
#        pprime[i] = derivpoly(coefficients,r[i])
    accuracy = 1e-8 # minimum accuracy
    for i in range(N): #perform newton's method
        temp[i]=newton(coefficients,r[i])
        if (temp[i] - r[i]) > accuracy:
            r[i]=temp[i]
        #else: #for checking to see the convergence
            #print("Calculating ... ",-temp[i], "meters from Earth ... ") 
    print("The Lagrange point, L1, is {:6.0f} meters from the surface\nof the Earth, between the Earth and the Moon.".format(-temp[i]))
main_end_time = time.time()

# Graphing and Visualization
graph_start_time = time.time()
#scale_earth_radius = 6371e3/3.844e8
#scale_moon_radius = 1737e3/3.844e8
#scale_L1_position = (-temp[i]+6371e3)/(3.844e8-1737e3)
#print(scale_earth_radius,scale_moon_radius, scale_L1_position)
#matplotlib.pyplot.plot(r,p,"k-",label="GMr^-2 - Gm(R-r)^-2 - rw^2 = 0")
#matplotlib.pyplot.plot(r,pprime,"k--",label="derivative")
#ax = matplotlib.pylab.subplot(111, polar=True)
#earthcircle = matplotlib.pylab.Circle((0,0), scale_earth_radius, transform=ax.transData._b, color="green", alpha=1.0)
#mooncircle = matplotlib.pylab.Circle((1,0), 5*scale_moon_radius, transform=ax.transData._b, color="gray", alpha=1.0)
#L1circle = matplotlib.pylab.Circle((scale_L1_position,0), 0.01, transform=ax.transData._b, color="black", alpha=1.0)
#ax.add_artist(earthcircle)
#ax.add_artist(mooncircle)
#ax.add_artist(L1circle)
end_time = time.time()

# Timing
#    print("----------------\nTime\n----------------")
#    print("imports . . . . . . {:4.2f} ms".format(import_end_time-import_start_time))
#    print("definitions . . . . {:4.2f} ms".format(def_end_time-def_start_time))
#    print("main calculation  . {:4.2f} ms".format(main_end_time-main_start_time))
#    print("graphs  . . . . . . {:4.2f} ms".format(end_time-graph_start_time))
#    print("Total . . . . . . . {:4.2f} ms\n----------------".format(end_time-start_time))
#print("To finish the program, close the graph window.")
#matplotlib.pyplot.show()
#matplotlib.pylab.show()
print("----------------\nThanks! -- END PROGRAM")

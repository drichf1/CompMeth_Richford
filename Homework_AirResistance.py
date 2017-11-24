print("This program is for Homework 5: ODEs.")
print("-------------------------------------")
print("This program calculates the trajectory\n \
of a projectile of varying mass and\n \
graphs the trajectories and the final\n \
velocities upon impact.")

import numpy # pi, square root, array, linspace, zeroes, cosine, sine
import matplotlib.pyplot # plot, show, scatter, subplots

class rk4_solver:
    def __init__(self,f): #initialization, takes function, f
        self.f = f #make function an attribute f the class
        self.initial_conditions = None #initial conditions
        self.solution = None #solution
    def solve(self,a,b,N=1000): #only need to call a (lower lim), b (upper lim)
        #print("RK4 Solver")
        # set up
        f = self.f # the function we want to solve
        r0 = numpy.array(self.initial_conditions,float) #init cond.
        h = (b-a)/N # interval spacing
        tpoints = numpy.linspace(a,b,N) #time points
        solution = numpy.zeros(tpoints.shape + r0.shape,float) #solutions
        # do the solving
        r = r0 # initialize things
        #print("r",r)
        for i,t in enumerate(tpoints):
            solution[i]=r
            k1 = h*f(r,t)
            #print(h*f(r,t))
            k2 = h*f(r+0.5*k1,t+0.5*h)
            k3 = h*f(r+0.5*k2,t+0.5*h)
            k4 = h*f(r+k3,t+h)
            r += (k1+2*k2+2*k3+k4)/6
            #print("RK4 says, 'Hi!'")
        # more attrtibutes of the class pertinent to the iteration
        self.h = h # make spacing an attribute of the class
        self.t = tpoints #make tpoints an attribute, too
        # record solution
        self.solution = solution #make the solution an attribute of the class

# I also wanted to try to make things all in a single function, with subfunctions
def get_trajectory(mass,left): # takes a mass, since the problem asks for a mass range
    # Phsyical constants/etc.
    array = numpy.zeros(6)
    rho = 1.22 # kilograms per cubic meter (density of air)
    Cd = 0.47 # coeff of drag
    radius = 0.08 # cm (radius of ball)
    m = mass # kg (mass of ball) (argument of function)
    g = 9.81 # m/s^2 accel b/c of gravity

    # Initial Conditoiins & array
    v0 = 100. # m/s Initial Velocity
    theta0 = numpy.pi/6 # 30 degrees initial angle
    x0 = y0 = 0.0 # m, initial position
    vx0 = v0*numpy.cos(theta0) #initial x velocity
    vy0 = v0*numpy.sin(theta0) # initial y velocity
    r0 = [x0,y0,vx0,vy0] #array of initial values

    # Constant for use in differential Equation
    constant = 0.5*numpy.pi*radius*radius*Cd*rho/mass

    #define function we're checking
    def drag_diffeq(r,t):
        x,y,vx,vy = r
        v = numpy.sqrt(vx*vx+vy*vy)
        deriv_r = [vx,vy] #derivative of position is velocity (2 eqns)
        deriv_v = [-constant*vx*v, -constant*vy*v-g] #der iv of velocity (2 eqns)
        return numpy.array(deriv_r + deriv_v) #joins the arrays together

    # do the program
    drag = rk4_solver(drag_diffeq) #make a solver object for the drag eqns
    drag.initial_conditions = r0 #set initial conditions
    drag.solve(0,10, 100) # solve for the time intervals 0s, 10s; 1000 iterations; omit "self" arg
    x = drag.solution[:,0]
    y = drag.solution[:,1]
    vxend = drag.solution[-1,2]
    vyend = drag.solution[-1,3]
    # returns
    left = matplotlib.pyplot.plot(x[y>0],y[y>0],label=mass)
    return numpy.sqrt(vxend*vxend+vyend*vyend)

if __name__ == "__main__":
    mass_range = numpy.linspace(1,10,10)
    # setting up the graphs
    f, (left, right) = matplotlib.pyplot.subplots(1, 2)
    left.set_title("Trajectory")
    left.set_xlabel("Horizontal Displacement (meters)")
    left.set_ylabel("Vertical Displacement (meters)")
    left.legend()
    right.set_title("Velocity upon Impact")
    right.set_xlabel("Mass of Projectile (kilograms)")
    right.set_ylabel("Velocity (meters per second)")
    # Calculate the RK4 Solution
    drop_speed_range = [get_trajectory(mass,left) for mass in mass_range]
    # Finish graphing
    right = matplotlib.pyplot.scatter(mass_range,drop_speed_range)
    print("---\nTo finish the program, close the graph window.")
    matplotlib.pyplot.show()

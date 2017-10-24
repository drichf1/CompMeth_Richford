# 0. Greeting
print("This is a program to calculate the motion with air resistance of a cannonball of varying mass.")
# 1. IMPORT STATEMENTS
import time
starttime = time.time()
import numpy
import matplotlib.pylab
end_import_time = time.time()
# 2. DEFINITIONS
# 2.1 MISC.
def distance(x=0.0,y=0.0,z=0.0):
    return numpy.sqrt(x*x+y*y+z*z)
def air_friction_coefficient(R,rho,C): # radius of obj, density of fluid, drag constant,
    return 0.5*numpy.pi*R*R*rho*C
def airfriction_force(A,v): # constant from drag eqn, velocity of particle
    return A*v*v
#2.2 DERIVATIVES
def horizontal_velocity_derivative(m,A,vy,t):
    return -(airfriction_force(A,vy)/m)-9.81
def vertical_velocity_derivative(m,A,vx,t): 
    return -(airfriction_force(A,vx)/m)
def horizontal_position_derivative(A, vx, vy,t): #constant from drag eqn, y-dot, x-dot
    return -A*vx*distance(vx,vy)
def vertical_position_derivative(A, vx, vy,t): #constant from drag eqn, grav accel, y-dot, x-dot
    return -9.81 - A*vy*distance(vx,vy)
end_def_time = time.time()
# 3. MAIN LOOP
if __name__ == "__main__":
    # 3.1 Initialize Parameters & Generate Constants
    R = 0.08 # m
    rho = 1.22 # kg m^-1
    C = 0.47
    A = air_friction_coefficient(R,rho,C)
    # 3.2 Initialize initial conditions and variables.0
    x0 = 0.0
    y0 = 0
    v0 = 100.0 # m/s
    vx0 = v0*numpy.cos(numpy.pi/6)
    vy0 = v0*numpy.sin(numpy.pi/6)
    mass = numpy.linspace(0.1,10,110)
    # 3.3 Start RK4 algorithm
    # 3.3.1 Step Size and Timesteps
    h = 0.001
    tpoints = numpy.arange(0,10,h)
    # 3.3.2 Empty Arrays of Arrays for Answers
    horizontal_velocity_derivative_steps_vx = []
    vertical_velocity_derivative_steps_vy = []
    horizontal_position_derivative_steps_x = []
    vertical_position_derivative_steps_y = []
    # 3.3.3 Loop Over Each Value of the Mass
    for i in range(len(mass)):
        # 3.3.3.1 Initial Conditions and Append New Empty Array
        vxout = vx0
        vyout = vy0
        horizontal_velocity_derivative_steps_vx.append([])
        vertical_velocity_derivative_steps_vy.append([])
        xout = x0
        yout = y0
        horizontal_position_derivative_steps_x.append([])
        vertical_position_derivative_steps_y.append([])
        # 3.3.3.2 RK4 Run Over Each Timestep
        for t in tpoints:
            # 3.3.3.2.1 Record Answer from Last Loop
            horizontal_velocity_derivative_steps_vx[i].append(vxout)
            vertical_velocity_derivative_steps_vy[i].append(vyout)
            horizontal_position_derivative_steps_x[i].append(xout)
            vertical_position_derivative_steps_y[i].append(yout)
            # 3.3.3.2.2 RK4 Coefficients for Horizontal Velocity
            k1vx = h*horizontal_velocity_derivative(mass[i],A,vxout,t)
            k2vx = h*horizontal_velocity_derivative(mass[i],A,vxout+0.5*k1vx,t+0.5*h)
            k3vx = h*horizontal_velocity_derivative(mass[i],A,vxout+0.5*k2vx,t+0.5*h)
            k4vx = h*horizontal_velocity_derivative(mass[i],A,vxout+k3vx,t+h)
            # 3.3.3.2.3 RK4 Coefficients for Vertical Velocity
            k1vy = h*vertical_velocity_derivative(mass[i],A,vyout,t)
            k2vy = h*vertical_velocity_derivative(mass[i],A,vyout+0.5*k1vy,t+0.5*h)
            k3vy = h*vertical_velocity_derivative(mass[i],A,vyout+0.5*k2vy,t+0.5*h)
            k4vy = h*vertical_velocity_derivative(mass[i],A,vyout+k3vy,t+h)
            # 3.3.3.2.4 RK4 Coefficients for Horizontal Position
            k1x = h*horizontal_position_derivative(A,xout,yout,t)
            k2x = h*horizontal_position_derivative(A,xout+0.5*k1x,yout,t+0.5*h)
            k3x = h*horizontal_position_derivative(A,xout+0.5*k2x,yout,t+0.5*h)
            k4x = h*horizontal_position_derivative(A,xout+k3x,yout,t+h)
            # 3.3.3.2.5 RK4 Coefficients for Vertical Position
            k1y = h*vertical_position_derivative(A,xout,yout,t)
            k2y = h*vertical_position_derivative(A,xout,yout+0.5*k1y,t+0.5*h)
            k3y = h*vertical_position_derivative(A,xout,yout+0.5*k2y,t+0.5*h)
            k4y = h*vertical_position_derivative(A,xout,yout+k3y,t+h)
            # 3.3.3.2.6 Increment the Result Parameters
            vxout += (k1vx+2*k2vx+2*k3vx+k4vx)/6
            vyout += (k1vy+2*k2vy+2*k3vy+k4vy)/6
            xout += (k1x+2*k2x+2*k3x+k4x)/6
            yout += (k1y+2*k2y+2*k3y+k4y)/6
    end_main_loop = time.time()
# 4. Graphing
    print("To continue to the timing analysis of the program,\nplease close the graphing window.")
# 5. Timing
# 6. Ending
    print("Thank you for using this program! Have a good day. ^-^")

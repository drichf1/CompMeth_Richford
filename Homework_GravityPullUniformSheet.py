# Gravity Integral
import time
start_time = time.time()
print("This is a tool to calculate the gravitational pull on a point mass at the center of a massive plane.\nThe calculation should take approximately 10 seconds and will generate a graph.")

import_start_time = time.time()
import numpy
import matplotlib.pylab

import_end_time = time.time()
def_start_time = time.time()

def gaussxw(N):
    # Initial approximation to roots of the Legendre polynomial
    a = numpy.linspace(3,4*N-1,N)/(4*N+2)
    x = numpy.cos(numpy.pi*a+1/(8*N*N*numpy.tan(a)))
    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = numpy.ones(N,float)
        p1 = numpy.copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))
    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)
    return x,w

def gaussxwab(N,a,b):
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w

def integrand(x,y,z):
    return (x**2 + y**2 + z**2)**-3/2

def force(z,lower_limit,upper_limit): #w,x,y arrays; z a value
    sigma = 10000/25
    m = 1
    N = 100
    x,w = gaussxw(N)
    x_points = 0.5*(upper_limit-lower_limit)*x + 0.5*(upper_limit+lower_limit)
    y_points = numpy.copy(x_points)
    w_points = 0.5*(upper_limit-lower_limit)*w
    G = 6.674e-11 # m**3 kg**-1 s**-2
    sum_result = 0.
    for i in range(N):
        for j in range(N):
            sum_result += w_points[i]*w_points[j] * integrand(x_points[i],y_points[j],z)
    return m*sigma*G*z*sum_result
def_end_time = time.time()

main_start_time=time.time()
if __name__ == "__main__":
    z_array = numpy.linspace(0,10,100)
    force_array_100 = [force(k,-100,100) for k in z_array]
    force_array_10 = [force(k,-10,10) for k in z_array]
    force_array_5 = [force(k,-5,5) for k in z_array]
    force_array_1 = [force(k,-1,1) for k in z_array]
    main_end_time = time.time()
    graph_start_time = time.time()
    plt1 = matplotlib.pyplot.plot(z_array,force_array_1, label="1m x 1m")
    plt5 = matplotlib.pyplot.plot(z_array,force_array_5, label="5m x 5m")
    plt10 = matplotlib.pyplot.plot(z_array,force_array_10, label="10m x 10m")
    plt100 = matplotlib.pyplot.plot(z_array,force_array_100, label="100m x 100m")
    matplotlib.pyplot.title("Gravitational Force on a Point Mass (1kg)\nfrom the Center of a Square Plane")
    matplotlib.pyplot.xlabel("Distance from Center of Pheet (meters)")
    matplotlib.pyplot.ylabel("Logarithm of\nGravitational Force (newtons)")
    matplotlib.pyplot.semilogy()
    matplotlib.pyplot.grid()
    matplotlib.pyplot.legend()
    end_time=time.time()
#    print("----------------\nTime\n----------------")
#    print("imports . . . . . . {:4.2} ms".format(import_end_time-import_start_time))
#    print("definitions . . . . {:4.2} ms".format(def_end_time-def_start_time))
#    print("main calculation  . {:4.2} ms".format(main_end_time-main_start_time))
#    print("graphs  . . . . . . {:4.2} ms".format(end_time-graph_start_time))
#    print("Total . . . . . . . {:4.2} ms\n----------------".format(end_time-start_time))
#    print("To exit, close the graph's pop-up window. The program will end.")
    matplotlib.pyplot.show()

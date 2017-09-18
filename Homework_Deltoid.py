# A Tool To Plot A Few Polar Graphs
import numpy
import matplotlib.pyplot
from numba import jit

#definitions
@jit
def deltoid(angle): # Takes the angle and parametrizes cartesian coordinates
    if ((angle < 0) or (angle > 2*numpy.pi)): # keep angles between [0,2pi]
        angle = angle%(2*numpy.pi) #modulo 2*pi
    x = 2.*numpy.cos(angle) + numpy.cos(2.*angle) #x parametrization
    y = 2.*numpy.sin(angle) + numpy.sin(2.*angle) #y parametrization
    return numpy.array([x,y]) #returns array of x, y

@jit
def galilean_spiral(angle): #r = theta^2
    radius = angle**2
    return radius #returns radius

@jit
def fey_s_function(angle):
    radius = numpy.exp(numpy.cos(angle)) - 2*numpy.cos(4*angle) + (numpy.sin(angle/12))**5
    return radius #returns radius

@jit
def polar_to_2d_cartesian(radius,angle): #tool to convert radius and angle into x and y coordinates
    x = radius*numpy.cos(angle)
    y = radius*numpy.sin(angle)
    return numpy.array([x,y]) #returns x, y array

if __name__ == "__main__": #if we're running this tool
    print("This is a tool to plot various polar curves.") # a little hello to the users
    # User Choice of Function
    user_input = input("Please select your curve.\nType 'deltoid,' 'galilean,' 'fey,' or 'all':  ")
    while ((user_input != 'deltoid') and (user_input != 'galilean') and (user_input != 'fey') and (user_input != 'all')):
        user_input = input("Please type only 'deltoid', 'galilean', 'fey', or 'all'.")

    #Start Calculations!
    angle_min, angle_max, angle_step = 0., 2*numpy.pi, 100 #details of how to choose the angles
    angles = numpy.linspace(angle_min, angle_max, angle_step) #make a lin-space array for them
    angle_index = 0 #initialize the index for the while loops
    # Deltoid
    if (user_input == 'deltoid'):
        print("Deltoid: looks like a calla lily!")       #Choose All if you're curious
        x = numpy.zeros(angles.shape,dtype=float)        #array of x-values, modeled on angle-array
        y = numpy.zeros(angles.shape,dtype=float)        #array of y-values, modeled on the angle-array
        while (angles[angle_index] < angle_max):         # interval [0,2pi)
            deltoid_array = deltoid(angles[angle_index]) #calculate the deltoid
            x[angle_index]=deltoid_array[0]              #record the x
            y[angle_index]=deltoid_array[1]              #record the y
            angle_index += 1                             #increment the index
        if (angles[angle_index] == angle_max):           # "manually" set x(2pi) = x(0), y(2pi) = y(0)
            x[angle_index]=x[0]                          #record the x
            y[angle_index]=y[0]                          #record the y
        matplotlib.pyplot.plot(x,y)                      #plot
        matplotlib.pyplot.xlabel("X")                    #label
        matplotlib.pyplot.ylabel("Y")                    #label
        matplotlib.pyplot.show()                         #show
    #Galilean Spiral
    elif (user_input == 'galilean'):
        print("Galilean Spiral: looks like a stem!")              #Choose all if you're curious
        x = numpy.zeros(angles.shape,dtype=float)                 #x-array, modeled on the angles array
        y = numpy.zeros(angles.shape,dtype=float)                 #y array 
        while (angles[angle_index] < angle_max):                  #interval [0,2pi)
            galileo_radius = galilean_spiral(angles[angle_index]) #make the spiral and get coordinates
            galileo_cartesian_coordinates = polar_to_2d_cartesian(galileo_radius,angles[angle_index])
            x[angle_index] = galileo_cartesian_coordinates[0]     #record the x
            y[angle_index] = galileo_cartesian_coordinates[1]     # record y
            angle_index += 1                                      #increment the index
        if (angles[angle_index] == angle_max):                    # "manually" set x(2pi), y(2pi)
            galileo_radius = galilean_spiral(angles[angle_index]) # make the spiral and get coordinates
            galileo_cartesian_coordinates = polar_to_2d_cartesian(galileo_radius,angles[angle_index])
            x[angle_index] = galileo_cartesian_coordinates[0]     #record x(2pi)
            y[angle_index] = galileo_cartesian_coordinates[1]     #record y(2pi)
        matplotlib.pyplot.plot(x,y)                               #plot
        matplotlib.pyplot.xlabel("X")                             #label
        matplotlib.pyplot.ylabel("Y")                             #label
        matplotlib.pyplot.show()                                  #show
    #Fey's Function
    elif (user_input == 'fey'):
        print("Fey's Function: looks like a butterfly!")     # Choose all if you're curious
        x = numpy.zeros(angles.shape,dtype=float)            #x-array matching angles array
        y = numpy.zeros(angles.shape,dtype=float)            #y-array matching angles array
        while (angles[angle_index] < angle_max):             #interval [0,2pi)
            fey_radius = fey_s_function(angles[angle_index]) #do fey's function and get coordinates
            fey_cartesian_coordinates = polar_to_2d_cartesian(fey_radius,angles[angle_index])
            x[angle_index] = fey_cartesian_coordinates[0]    #record x
            y[angle_index] = fey_cartesian_coordinates[1]    #record y
            angle_index += 1                                 #increment index
        if (angles[angle_index] == angle_max):               # "manually" do x(2pi),y(2pi)
            fey_radius = fey_s_function(angles[angle_index]) #calculate fey's function and coordinates
            fey_cartesian_coordinates = polar_to_2d_cartesian(fey_radius,angles[angle_index])
            x[angle_index] = fey_cartesian_coordinates[0]    #record x(2pi)
            y[angle_index] = fey_cartesian_coordinates[1]    #record y(2pi)
        matplotlib.pyplot.plot(x,y)                          #plot
        matplotlib.pyplot.xlabel("X")                        #label
        matplotlib.pyplot.ylabel("Y")                        #label
        matplotlib.pyplot.show()                             #show
    elif (user_input == "all"):
        print("Deltoid: looks like a calla lily!")                    #it makes a nice picture
        print("Galilean Spiral: looks like a stem!")
        print("Fey's Function: looks like a butterfly!")
        x_deltoid = numpy.zeros(angles.shape,dtype=float)             #Different x,y arrays for each function
        y_deltoid = numpy.zeros(angles.shape,dtype=float)
        x_galilean = numpy.zeros(angles.shape,dtype=float)
        y_galilean = numpy.zeros(angles.shape,dtype=float)
        x_fey = numpy.zeros(angles.shape,dtype=float)
        y_fey = numpy.zeros(angles.shape,dtype=float)
        while (angles[angle_index] < angle_max):                       #in the 0 to less-than-2pi interval
            deltoid_array = deltoid(angles[angle_index])               #calculate deltoid
            x_deltoid[angle_index]=deltoid_array[0]                    #record deltoid x
            y_deltoid[angle_index]=deltoid_array[1]                    #record deltoid y
            galileo_radius = galilean_spiral(angles[angle_index])      #calculate galileo
            galileo_cartesian_coordinates = polar_to_2d_cartesian(galileo_radius,angles[angle_index])
            x_galilean[angle_index] = galileo_cartesian_coordinates[0] #record galileo x
            y_galilean[angle_index] = galileo_cartesian_coordinates[1] #record galileo y
            fey_radius = fey_s_function(angles[angle_index])           #calculate fey
            fey_cartesian_coordinates = polar_to_2d_cartesian(fey_radius,angles[angle_index])
            x_fey[angle_index] = fey_cartesian_coordinates[0]          #record fey x
            y_fey[angle_index] = fey_cartesian_coordinates[1]          #record fey y
            angle_index += 1                                           #increment index
        if (angles[angle_index] == angle_max):                         # "manually" do the values for theta=2pi
            deltoid_array = deltoid(angles[angle_index])
            x_deltoid[angle_index]=deltoid_array[0]
            y_deltoid[angle_index]=deltoid_array[1]
            galileo_radius = galilean_spiral(angles[angle_index])
            galileo_cartesian_coordinates = polar_to_2d_cartesian(galileo_radius,angles[angle_index])
            x_galilean[angle_index] = galileo_cartesian_coordinates[0]
            y_galilean[angle_index] = galileo_cartesian_coordinates[1]
            fey_radius = fey_s_function(angles[angle_index])
            fey_cartesian_coordinates = polar_to_2d_cartesian(fey_radius,angles[angle_index])
            x_fey[angle_index] = fey_cartesian_coordinates[0]
            y_fey[angle_index] = fey_cartesian_coordinates[1]
        matplotlib.pyplot.plot(x_deltoid,y_deltoid,'b')                #plotting
        matplotlib.pyplot.plot(x_galilean,y_galilean,'g')
        matplotlib.pyplot.plot(x_fey,y_fey,'r')
        matplotlib.pyplot.xlabel("X")
        matplotlib.pyplot.ylabel("Y")
        matplotlib.pyplot.show()

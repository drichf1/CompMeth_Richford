# 4.1 Floating Factorial Problem
# Dan Richford
print("This is a tool to calculate the factorials of integers and floating point numbers -- and to calculate their differences.")
#imports
import numpy
import matplotlib.pyplot
#definitions
def integer_factorial(number): # Takes an integer and calculates the factorial
    if (type(number) == int): # check & calculate the factorial
        f = 1
        for k in range(1, number + 1):
            f *= k
        return f
    else:
        return -1

def float_factorial(number):
    if (type(number) == float):
        f = 1.0
        for k in range(1, int(number)+1): #range numbers must be integers
            f *= float(k)
        return f
    else:
        return -1.0
#Run!
if __name__ == "__main__": # If running in the terminal
# Use these lines if you want the user to specify the inputs
#    integer_number = int(input("Please enter an integer: "))
#    floating_point_number = float(integer_number)
# A loop to plot the differences between float factorial and integer factorials
    index = 0 #index for loop
    x = [] # List that will contain the input integers
    y = [] # List that will contain the differences
    while index <= 155: #155 chosen because 156+ result in overflow errors and defining factorial to be infinity
        integer_number = int(index) #extract integer
        floating_point_number = float(index) #extract float
        fact_int = integer_factorial(integer_number) #calculate factorial integer
        fact_flt = float_factorial(floating_point_number) #calculate float factorial
        difference_from_integer = numpy.abs(fact_int + (-1)*fact_flt) #calculate difference
        if (difference_from_integer > 0): #if difference is greater than zero, record the input and difference
            x.append(index)
            y.append(difference_from_integer)
# Explicit print statements for checking
#            print("Non-zero integer-float difference detected! {}".format(integer_number))
#            print("    The factorials of {} are:".format(integer_number))
#            print("     -- as the floating-point number {}: ".format(floating_point_number)," {}".format(fact_flt))
#            print("     -- as the integer number {}: ".format(integer_number)," {}".format(fact_int))
#            print("     The difference is {}".format(difference_from_integer),"\n")
        index += 1 #increment index
    #Plotting
    matplotlib.pyplot.plot(x,y)
    matplotlib.pyplot.title("Difference between Integer and Factorial")
    matplotlib.pyplot.semilogy()
    matplotlib.pyplot.ylabel("Difference")
    matplotlib.pyplot.xlabel("Number")
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.show()

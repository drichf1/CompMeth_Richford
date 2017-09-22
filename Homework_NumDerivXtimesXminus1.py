#4.3 calculating derivatives
import numpy
import matplotlib.pyplot
def numerical_forward_derivative(number,spacing):
    x = number
    h = spacing
    fd = ((x+h)*((x+h)+(-1)) + (-1)*(x*(x-1)))*(h**(-1))
    return fd
def numerical_backward_derivative(number, spacing):
    x = number
    h = spacing
    bd = ((-1)*((x-h)*((x-h)+(-1))) + (x*(x-1)))*(h**(-1))
    return bd
def numerical_central_difference_derivative(number, spacing):
    x = number
    h = spacing
    cd = ((-1)*((x-(h*0.5))*(x-(h*0.5)+(-1))) + ((x+(h*0.5))*((x+(h*0.5))+(-1))))*(h**(-1))
    return cd
def test_function(x):
    return x*(x-1)

if __name__ == "__main__":
    number = 1
    index = 1
    space_list = []
    analytical_list = []
    fd_list = []
    bd_list = []
    cd_list = []
    while ((index > 0) and (index <= 7)):
        spacing = 10**((-1)*(2*index))
        analytical = 1
        fd = numerical_forward_derivative(number,spacing)
        bd = numerical_backward_derivative(number, spacing)
        cd = numerical_central_difference_derivative(number, spacing)
        print("---\nThe spacing is {}.".format(spacing))
        print("The forward derivative of x(x-1) at x = ",number," is {:8.6}".format(fd))
        print("The backward derivative of x(x-1) at x = ",number," is {:8.6}".format(bd))
        print("The central-difference derivative of x(x-1) at x = ",number," is {:8.6}".format(cd))
        space_list.append(spacing)
        analytical_list.append(analytical)
        fd_list.append(fd)
        bd_list.append(bd)
        cd_list.append(cd)
        index += 1
    matplotlib.pyplot.plot(space_list,fd_list,'r-')
    matplotlib.pyplot.plot(space_list,bd_list,'b-')
    matplotlib.pyplot.plot(space_list,cd_list,'g-')
    matplotlib.pyplot.plot(space_list,analytical_list,'k.')
    matplotlib.pyplot.semilogx()
    matplotlib.pyplot.show()

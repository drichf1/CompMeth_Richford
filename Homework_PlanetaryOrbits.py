# Daniel Richford
# Homework Exercise 2.6
# Kepler: l2*v2 = l1*v1
# Cons. of Energy: E = 1/2 m v**2 - GmM/r
#     Show v2 is the smaller root of the quadratic eqn
#
#     v2**2 + [-(2*G*M)/(l1*v1)]*v2 + [-(v1**2 - (2*G*M)/l1)] = 0
#
#   - I don't know how to show this using a computer:
#     analytically, I know that the quadratic equation ends up being:
#
#         v2 = 2GM     1   +   1       (4 G^2 M^2       [        2GM])
#              ---  x ---  or  - x sqrt(--------- - 4 x [-v1^2 - ---])
#              l1v1    2   -   2       (l1^2 v1^2       [         l1])
#
#            = GM    + 1   ()
#              ---- or - x ()
#              l1v1  - 2   ()
import numpy

def user_input(): # specify l1, v1 of satellite we're interested in
    input_string = input("Which satellite would you like?\nType 'halley,' 'earth,' or 'other': ")
    while (input_string != "halley") and (input_string != "earth") and (input_string != "other"):
        input_string = input("Please type only 'halley','earth', or 'other': ")
    if (input_string == "halley"):
        l1,v1=8.7830*10**10,5.4529*10**4
    elif (input_string == "earth"):
        l1,v1=1.4710*10**11,3.0287*10**4
    elif (input_string == "other"):
        l1 = float(input("Please specify the perihelion distance: "))
        v1 = float(input("Please specify the perihelion linear velocity: "))
    return numpy.array([l1,v1], float)

def quadratic_formula_coefficients(inputs): #takes inputs as an array [l1, v1]
    G = 6.67*10**(-11) #m**3/kg*s**2
    M = 1.9881*10**30 #kg
    a = 1.
    b = -(2.*G*M)/(inputs[0]*inputs[1])
    c = -(inputs[1]**2. - (2.*G*M)/inputs[0])
    return numpy.array([a,b,c], float)

def quadraticformula(coefficients): #takes coefficients as an array [a,b,c]
    added_root = (-coefficients[1]+numpy.sqrt(coefficients[1]**2.-4.*coefficients[0]*coefficients[2]))/(2.*coefficients[0])
    subtracted_root = (-coefficients[1]-numpy.sqrt(coefficients[1]**2.-4.*coefficients[0]*coefficients[2]))/(2.*coefficients[0])
    return numpy.array([added_root,subtracted_root], float)

# Use the proper Root & get v2
def linear_velocity_at_aphelion(inputs, radius, roots): #takes array for roots [add. root, sub. root], array for [l1, v1]
    #centripital force = gravitational force
    G = 6.67*10**(-11) #m**3/kg*s**2
    M = 1.9881*10**30 #kg
    circular_velocity = numpy.sqrt(M*G/inputs[0]) # circular orbit
    escape_velocity = numpy.sqrt(2*M*G/inputs[0]) # escape veloc.
    #Gut-check to make sure an orbit occurs
    #Check to make sure our perihelion distance (l1) is greater than the radius of the sun
    if (inputs[0] <= radius):
        print("Oh, no! THe satellite will crash!")
        return -1.
    #check to make sure that the velocity at perihelion (v1) is less than the escape velocity
    if (inputs[1] >= escape_velocity):
        print("Oh, no! The satellite will not make an orbit!")
        return -1.
    #Analyze Orbit to determine v2 from roots
    #check to make sure orbit is elliptical
    if (inputs[1] == circular_velocity):
        print("Oh-ho! Orbit is circular!")
        return -1.
    #check if we're at perihelion -> use subtractive root
    elif (inputs[1] < circular_velocity):
        print("aphelion!")
        v2 = roots[0]
        return v2
    #check if we're at aphelion -> use additive root
    elif (inputs[1] > circular_velocity):
        print("perihelion!")
        v2 = roots[1]
        return v2

# original idea: use one definition -- I prefer to do in stages and print()
#def linear_velocity_at_aphelion(inputs): #takes inputs as an array [l1, v1]
#    G = 6.67*10**(-11) #m**3/kg*s**2
#    M = 1.9881*10**30 #kg
#    r = 1.*10**6
#    coefficients = quadratic_formula_coefficients(inputs)
#    roots = quadraticformula(coefficients)
#    v2 = checkroots(inputs, r, roots)
#    return v2

def aphelion(inputs,v2): #takes inputs as array [l1,v1]
    l2 = (inputs[0]*inputs[1])/v2
    return l2

def semimajor_axis(inputs, l2): #takes inputs as an array [l1, v1]
    a = (0.5)*(inputs[0]+l2)
    return a

def semiminor_axis(inputs, l2): #takes inputs as an array [l1, v1]
    b = numpy.sqrt(inputs[0]*l2)
    return b

def orbital_period(a,b,inputs): #takes inputs as an array [l1, v1]
    T = (2.*numpy.pi*a*b)/(inputs[0]*inputs[1])
    return T

def orbital_eccentricity(inputs, l2): #takes inputs as an array [l1, v1]
    e = (l2-inputs[0])/(l2+inputs[0])
    return e


#radius of sun
radius = 695.7e6 #meters
print("radius of sun")
print(radius)

#get user input
inputs = user_input()
print("you chose:")
print("[ l1, v1 ]")
print(inputs)

#calculate roots
coefficients = quadratic_formula_coefficients(inputs)
roots = quadraticformula(coefficients)
print("roots")
print("[ additive root, subtractive root ]")
print(roots)

#check roots and get linear velocity at aphelion
v2 = linear_velocity_at_aphelion(inputs, radius, roots)
print("v2")
print(v2)

#calculate aphelion
l2 = aphelion(inputs, v2)
print("l2")
print(l2)

#calculate semimajor axis, semiminor axis, oribital period, eccentricity
semimajor = semimajor_axis(inputs, l2)
semiminor = semiminor_axis(inputs, l2)
period = orbital_period(semimajor, semiminor, inputs)
eccentricity = orbital_eccentricity(inputs,l2)
print("Semimajor Axis: ", semimajor)
print("Semiminor Axis: ", semiminor)
print("Orb. Per.: ", period)
print("Orb. Ecc.: ", eccentricity)

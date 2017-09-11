# Semi-Empirical Mass Formula
# Daniel Richford

import numpy

# Program to calculate the binding energy based on values of atomic number (Z) and mass number (A).
def calculate_binding_energy(atomic_number, mass_number):
    #array for results
    energy_array = numpy.zeros((2,), dtype=numpy.float)
    #constants
    a_1 = 15.8 
    a_2 = 18.3
    a_3 = 0.714
    a_4 = 23.2
    if (mass_number%2 != 0): #if A is odd
        a_5 = 0
    elif ((mass_number%2 == 0) and (atomic_number%2 == 0)): #if both A,Z are even
        a_5 = 12.0
    elif ((mass_number%2 == 0) and (atomic_number%2 != 0)): #if A is even and Z is odd
        a_5 = -12.0
    #Calculate Binding Energy
    binding_energy = a_1*mass_number - a_2*numpy.cbrt(mass_number)**2. - a_3*((atomic_number**2)/(numpy.cbrt(mass_number))) - a_4*(((mass_number-2.*atomic_number)**2)/mass_number) + a_5/(numpy.sqrt(mass_number))
    #Calculate Binding Energy per Nucleon 
    binding_energy_per_nucleon = binding_energy/mass_number
    #fill array
    energy_array[0] = binding_energy
    energy_array[1] = binding_energy_per_nucleon
    return energy_array

# (2.10.a, .b) Print out the energy of Z=28,A=58
print("(a)")
array = numpy.zeros(2) #array to hold our energies
array = calculate_binding_energy(28,58)
print("Binding Energy: ", array[0], " MeV")
print("(b)")
print("Binding Energy per Nucleon: ", array[1], " MeV")

# (2.10.c) Cycle Through mass numbers
print("(c)")
atomic_number_z = int(input("specify atomic number: Z = "))
mass_number_a = atomic_number_z #index for while loop -- starts at only-proton nucleus
mass_number_store_a = 0 # keep track of largest mass number
energy_array_old = numpy.zeros(2) # holds last energy array
energy_array_new = numpy.zeros(2) #holds new energy array
for mass_number_a in range(atomic_number_z <= 3*atomic_number_z): # loop through range A=Z to A=3Z
    mass_number_a += 1 #get mass counter going (starts at 1)
    energy_array_new = calculate_binding_energy(atomic_number_z, mass_number_a)
    if (atomic_number_z == 0 and mass_number_a == 1): #seperately account for neutron alone
        energy_array_old = energy_array_new
        mass_number_store_a = mass_number_a
    elif (energy_array_old[1] < energy_array_new[1]):
        energy_array_old = energy_array_new
        mass_number_store_a = mass_number_a
        #print("Z = ",atomic_number_z,"\nA = ",mass_number_a)
        #print("Binding Energy per Nucleon: ",energy_array_old[1], " MeV")
print("Most Stable Isotope is: A = ",mass_number_store_a,", at a Binding Energy per Nucleon: ",energy_array_old[1], " MeV")

# (2.10.d) Cycle through first 100 elements
print("(d)")
atomic_number_z = 0 #index for the loop 
mass_number_a = 0 #index for the loop
atomic_number_store = 0 # variable to store the max value
mass_number_store = 0 # variable to store the max value
energy_array_old = numpy.zeros(2) # holds last energy array
energy_array_new = numpy.zeros(2) #holds new energy array
for atomic_number_z in range(1,118):
    atomic_number_z +=1
    mass_number_a = atomic_number_z
    for mass_number_a in range(atomic_number_z,3*atomic_number_z):
        mass_number_a += 1
        energy_array_new = calculate_binding_energy(atomic_number_z, mass_number_a)
        if (atomic_number_z == 0 and mass_number_a == 1): #seperately account for neutron alone
            energy_array_old = energy_array_new
            mass_number_store = mass_number_a
        #print("Z = ",atomic_number_z,"\nA = ",mass_number_a)
        #print("Binding Energy per Nucleon: ",energy_array_new[1])
        elif (energy_array_old[1] < energy_array_new[1]):
            energy_array_old = energy_array_new
            atomic_number_store = atomic_number_z
            mass_number_store = mass_number_a
            #print("    Swap!\n        Z = ",atomic_number_z,"\n        A = ",mass_number_a)
            #print("        Binding Energy per Nucleon: ",energy_array_old[1])
print("Most Stable Isotope is: Z = ",atomic_number_store," A = ",mass_number_store,", at a Binding Energy per Nucleon: ",energy_array_old[1], " MeV")

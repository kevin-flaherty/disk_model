# Define the Disk class. This class of objects can be used to create a disk structure. Given parameters defining the disk, it calculates the dust density structure using a simple radial power-law relation and defines the grid used for radiative transfer. This object can then be fed into the modelling code which does the radiative transfer given this structure.

#two methods for creating an instance of this class

# from disk import *
# x=Disk()

# import disk
# x = disk.Disk()

# For testing purposes use the second method. Then I can use reload(disk) after updating the code
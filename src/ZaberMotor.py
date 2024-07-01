"""
A module for controlling the Zaber motors: LAC10A-T4A (through a X-MCC), X-LSM and X-LHM

Need to come up with a way to be able to name an axis/optic and move the right controller
Ideas:
- XMCC class with usage like XMCC[<axis number>].move_absolute(1000), + a dictionary that maps 
    the name of the optic to both the axis number and controller
"""

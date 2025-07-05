import numpy as np
import sys
import glob
import os
from xaosim.shmlib import shm


class dmclass():
    """wrapper of Frantz shm specifically for control of Asgard's DM's"""
    def __init__(self, beam_id, shape_wdir='',main_chn = 2):
        
        beam_id = int(beam_id)

        assert beam_id in [1,2,3,4]
        # beam number 
        self.beam_id = beam_id
        # where DM shapes are kept
        self.shape_wdir = shape_wdir 
        # sub channels shared memory 
        self.shmfs = np.sort(glob.glob(f"/dev/shm/dm{beam_id}disp*.im.shm"))
        #combined channels 
        self.shmf0 = f"/dev/shm/dm{beam_id}.im.shm"
        # number of sub channels
        self.nch = len(self.shmfs)
        # main channel to apply DM commands to
        self.main_chn = main_chn
        # actual shared memory objects 
        self.shms = []
        for ii in range(self.nch):
            self.shms.append(shm(self.shmfs[ii],nosem=False))
            print(f"added: {self.shmfs[ii]}") 
        #actual combined shared memory 
        if self.nch != 0:
            self.shm0 = shm(self.shmf0, nosem=False)
        else:
            print("Shared memory structures unavailable. DM server started?")
            
            

    def select_flat_cmd(self,  wdir='DMShapes'):
        '''Matches a DM flat command file to a DM id #.

        Returns the name of the file in the work directory.
        '''
        flat_cmd_files = {
                        "1":"17DW019#113_FLAT_MAP_COMMANDS.txt",
                        "2":"17DW019#053_FLAT_MAP_COMMANDS.txt",
                        "3":"17DW019#093_FLAT_MAP_COMMANDS.txt",
                        "4":"17DW019#122_FLAT_MAP_COMMANDS.txt"
                        }
        
        return wdir + '/' + flat_cmd_files[f"{self.beam_id}"]


    def select_flat_cmd_offset(self,  wdir='DMShapes'):
            '''Matches a DM flat command file to a DM id #.

            Returns the name of the most recent file in the work directory for each beam.
            '''
            # flatoffset_cmd_files = {
            #                 "1":"BEAM1_FLAT_MAP_OFFSETS.txt",
            #                 "2":"BEAM2_FLAT_MAP_OFFSETS.txt",
            #                 "3":"BEAM3_FLAT_MAP_OFFSETS.txt",
            #                 "4":"BEAM4_FLAT_MAP_OFFSETS.txt"
            #                 }
            flatoffset_cmd_files = {}
            for beam in ["1", "2", "3", "4"]:
                pattern = os.path.join(wdir, f"BEAM{beam}_FLAT_MAP*.txt")
                matching_files = glob.glob(pattern)
                
                if not matching_files:
                    flatoffset_cmd_files[beam] = None  # or raise an error / warning
                else:
                    # Get the most recent file by modification time
                    latest_file = max(matching_files, key=os.path.getmtime)
                    flatoffset_cmd_files[beam] = latest_file
                    
            return flatoffset_cmd_files[f"{self.beam_id}"]

    def cmd_2_map2D(self, cmd, fill=np.nan):
        '''Convert a 140 cmd into a 2D DM map for display.

        shm set_data method requires 2D 144 array?

        Just need to add the four corners (0 or nan) and reshape
        Parameters:
        - cmd  : 1D numpy array of 139 components
        - fill : filling values for corners (default = np.nan)
        '''
        return np.insert(cmd, [0, 10, 130, 140], fill).reshape((12, 12))

    def activate_flat(self):
        """
        convention to apply flat command on channel 0!
        """
        if self.nch == 0:
            return
        wdir = "/home/asg/Progs/repos/asgard-alignment/DMShapes/" #os.path.dirname(__file__)
        flat_cmd = np.loadtxt(self.select_flat_cmd( wdir))
        self.shms[0].set_data(self.cmd_2_map2D(flat_cmd, fill=0.0))
        ##
        self.shm0.post_sems(1)


    def activate_calibrated_flat(self):
        """
        convention to apply flat command on channel 0!
        this adds an additional BALDR calibrated offset to the DM flat
        """
        if self.nch == 0:
            return
        wdir = "/home/asg/Progs/repos/asgard-alignment/DMShapes/" #os.path.dirname(__file__)
        flat_cmd = np.loadtxt(self.select_flat_cmd( wdir))
        flat_cmd_offset = np.loadtxt(self.select_flat_cmd_offset( wdir))
        self.shms[0].set_data(self.cmd_2_map2D(flat_cmd + flat_cmd_offset, fill=0.0))
        ##
        self.shm0.post_sems(1)
        
        
    def get_baldr_flat_offset(self):
        # baldr calibrated offset from the BMC factory flat
        wdir = "/home/asg/Progs/repos/asgard-alignment/DMShapes/"
        flat_cmd_offset = np.loadtxt(self.select_flat_cmd_offset( wdir))
        return flat_cmd_offset 

    def activate_cross(self, amp=0.1):
        """
        convention to apply calibration shapes on channel 1!
        """
        dms=12
        ii0 = dms // 2 - 1 
        cross_cmd = np.zeros((dms,dms))
        cross_cmd[ii0:ii0+2, :] = amp
        cross_cmd[:, ii0:ii0+2] = amp
        self.shms[1].set_data(cross_cmd)
        ##
        self.shm0.post_sems(1)

    def apply_modes(self, amplitude_list, basis_list):
        """
        convention to apply DM modes on channel 2!
        amplitude_list is list of amplitudes to be applied to each mode in basis_list
        amplitude_list must be same lengthh as basis_list
        applies the amplitude weighted sum of modes to DM on shm channel 2 
        """        
        cmd = np.sum( [ aa * MM for aa, MM in zip(amplitude_list, basis_list)])
        self.shms[self.main_chn].set_data(cmd)
        ##
        self.shm0.post_sems(1)


    def set_data(self, cmd):
        """
        convention to apply any user specific commands on channel 2!
        """
        self.shms[self.main_chn].set_data(cmd)
        ##
        self.shm0.post_sems(1)

    def zero_all(self):
        cmd = np.zeros(144)
        for ii, ss in enumerate(self.shms):
            ss.set_data(cmd)
            
            print(f"zero'd {self.shmfs[ii]}")
        ## 
        self.shm0.post_sems(1)


    def close(self, erase_file = False):
        # freeing all shared memory structures
        for ii in range(self.nch):
            self.shms[ii].close(erase_file=erase_file)
        for ii in range(self.nch):
            self.shms.pop(0)
        print("end of program")
        #sys.exit()


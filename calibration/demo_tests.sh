# TT 
# 2nm RMS @ 2Hz
./common/TTdisturb_all.sh --dt 0.05 --dm_rms 0.02 --dist_type vib --vib_freq 2
# 50nm RMS @ 15Hz

./common/TTdisturb_all.sh --dt 0.005 --dm_rms 0.008 --dist_type vib --vib_freq 15
# ./playground/baldr_CL/close_all.sh --ki_LO 0.4 --kp_LO 0.2

# cpp rtc 
#update_pid_param ["LO","ki","all",0.97] # 
#update_pid_param ["LO","kp","all",0.4]  

# 15nm RMS @ 50Hz
./common/TTdisturb_all.sh --dt 0.001 --dm_rms 0.002 --dist_type vib --vib_freq 50

# HO 
# 100nm RMS (65% H strehl)
./common/turb_all.sh --V 1.2 --r0 0.4 --DM_chn 3 --number_of_modes_removed 0
# rtc cpp

#update_pid_param ["LO","ki","all",0.97]
#update_pid_param ["HO","ki","all",0.4]

# 200nm RMS (40% H strehl)
./common/turb_all.sh --V 1.2 --r0 0.2 --DM_chn 3 --number_of_modes_removed 6
# 350nm RMS (20% H strehl)
./common/turb_all.sh --V 1.2 --r0 0.15 --DM_chn 3 --number_of_modes_removed 6


# SNR 
# 200 Hz 

# 500 Hz 

# 1000 Hz 
# ./playground/baldr_CL/close_all.sh --ki_LO 0.4 --ki_HO 0.2


# generally Use 
# ./playground/baldr_CL/close_all.sh --ki_LO 0.6 --ki_HO 0.4



# when going faint with python rtc use Ki_LO = 0.4-0.5, ki_HO 0.1
#./playground/baldr_CL/close_all.sh --ki_LO 0.5 --ki_HO 0.1




### Day 2 

# TT 

# 140 nm RMS at 15Hz 
./common/TTdisturb_all.sh --dt 0.005 --dm_rms 0.02 --dist_type vib --vib_freq 15 
update_pid_param ["LO","ki","all",0.8]

# Piston
./common/pist_disturb_all.sh --dist_type vib --dt 0.005 --vib_freq 50 --dm_rms 0.1
update_pid_param ["LO","ki","all",0.8]


# HO 
./common/turb_all.sh --V 1.2 --r0 0.4 --DM_chn 3 --number_of_modes_removed 0
update_pid_param ["HO","ki","all",0.3]






# 140 nm RMS at 50Hz 
./common/TTdisturb_all.sh --dt 0.001 --dm_rms 0.02 --dist_type vib --vib_freq 50 
update_pid_param ["LO","ki","all",0.8]

# 1/f ~ 200nm RMS 
./common/TTdisturb_all.sh --dt 0.001 --dm_rms 0.03 --dist_type 1onf 
update_pid_param ["LO","ki","all",0.8]


# HO 
# 100nm RMS (65% H strehl)
./common/turb_all.sh --V 1.2 --r0 0.4 --DM_chn 3 --number_of_modes_removed 0
# update_pid_param ["LO","ki","all",0.02]
update_pid_param ["HO","ki","all",0.2]

# 200nm RMS (40% H strehl)
./common/turb_all.sh --V 1.2 --r0 0.3 --DM_chn 3 --number_of_modes_removed 0
update_pid_param ["LO","ki","all",0.05]
update_pid_param ["HO","ki","all",0.2]

# 350nm RMS (20% H strehl)
./common/turb_all.sh --V 1.2 --r0 0.15 --DM_chn 3 --number_of_modes_removed 0
update_pid_param ["HO","ki","all",0.1]
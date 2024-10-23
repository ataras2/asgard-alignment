import numpy as np
np.set_printoptions(precision=5, suppress=True)

dps = [835,966,1077,1194]

ii = 1043 #pixels per degree of beam motion (NB not mirror)
ip = 4.73 #mm per degree of beam motion

for i, dp in enumerate(dps):
	pp = 0.0014*dp + 1.94
	pi = 0.522*dp
	T = np.array([[pp, ip],[pi, ii]]) * 2
	print(f'Pupil and Image Matrix for beam {i+1}')
	print(np.linalg.inv(T))
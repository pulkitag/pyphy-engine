import numpy as np
import geometry as gm

def time_from_pt1_pt2(pt1, pt2, vel):
	displacement = pt2 - pt1
	velProj      = vel.project(displacement)
	s       = displacement.mag()
	v       = velProj.mag()
	if v == 0 and s==0:
		return 0
	elif v == 0:
		print "ZERO VELOCITY ENCOUNTERED"
		return np.inf
	else:
		return s/v

import numpy as np
import geometry as gm

def time_from_pt1_pt2(pt1, pt2, vel):
	displacement = pt2 - pt1
	velProj      = vel.project(displacement)
	s       = displacement.mag()
	v       = velProj.mag()
	return s/v

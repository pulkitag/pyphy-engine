import geometry as gm
import primitives as pm
import numpy as np

def get_toc_ball_wall(obj1, obj2):
	'''
		obj1: ball
		obj2: wall
	'''
	lines = obj2.get_lines()
	vel   = obj1.get_velocity()
	pos   = obj1.get_position()
	r     = obj1.get_radius()
	tCol    = np.inf
	nrmlCol = None
	for l in lines:
		#This is the nrml from the line towards the point
		nrml    = l.get_normal_towards_point(pos)
		nrml.scale(-1) #Normal from point towards line
		speed   = vel.dot(nrml)	
		if speed <=0:
			continue	
		#Distance from line
		ray = gm.Line(pos, pos + nrml)
		intPoint = l.get_intersection_ray(ray)
		assert intPoint is not None, "Intersection point cannot be none"
		dist     = pos.distance(intPoint)
		dist     = dist - r	
		assert dist >= 0, "Distance has to be >=0"
		t = dist / speed
		if t < tCol:
			tCol    = t
			nrml.scale(-1)
			nrmlCol = nrml
	return tCol, nrmlCol, intPoint 	 	
	
	

import geometry as gm
import primitives as pm
import numpy as np
import pdb

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
	ptCol   = None
	for l in lines:
		#This is the nrml from the line towards the point
		nrml    = l.get_normal_towards_point(pos)
		nrml.scale(-1) #Normal from point towards line
		speed   = vel.dot(nrml)	
		if speed <=0:
			continue
		#Velocity in orthogonal direction
		velOrth = vel - (speed * nrml)
		lDir    = l.get_direction()	
		#Find the time of collision
		ray = gm.Line(pos, pos + nrml)
		intPoint = l.get_intersection_ray(ray)
		assert intPoint is not None, "Intersection point cannot be none"
		distCenter = pos.distance(intPoint)
		dist       = distCenter - r	
		if dist < 0:
			#It is possible that a line (but not the line segment) intersects the ball. Then
			#dist < 0, and we need to rule out such cases. 
			assert distCenter >= 0
			onSegment = l.is_on_segment(intPoint)
			if onSegment:
				print "Something is amiss" 
				pdb.set_trace()
			else:
				continue
		t = dist / speed
		#Find the intersection point on line
		#i.e. the point 
		linePoint = intPoint + (t * velOrth)
		onSegment = l.is_on_segment(linePoint)
		if not onSegment:
			continue	
		if t < tCol:
			tCol    = t
			nrml.scale(-1)
			nrmlCol = nrml
			ptCol   = linePoint					
	
	#pdb.set_trace()
	print tCol, nrmlCol, ptCol
	return tCol, nrmlCol, ptCol	
	
	

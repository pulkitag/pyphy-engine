import geometry as gm
import primitives as pm
import numpy as np
import pdb

##
#Time of collision of a ball with a wall
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
	#print tCol, nrmlCol, ptCol
	return tCol, nrmlCol, ptCol	

##
#Time of collision of ball with ball. 	
def get_toc_ball_ball(obj1, obj2):
	#Initializations
	tCol    = np.inf
	nrmlCol = None
	ptCol   = None
	#Get the velocities of the ball. 
	pos1, vel1 = obj1.get_position(), obj1.get_velocity()
	pos2, vel2 = obj2.get_position(), obj2.get_velocity()
	#We will go into the frame of reference of object 1
	relVel = vel2 - vel1
	#Find the direction of collision
	colDir  = pos1 - pos2
	colDist = colDir.mag() - (obj1.get_radius() + obj2.get_radius())
	colDir.make_unit_norm() 
	#Get the velocity along the direction of collision
	speed = relVel.dot(colDir)
	if speed <= 0:
		#If the balls will not collide
		return tCol, nrmlCol, ptCol
	#There is one more situation in which no collision will happen.
	circ = gm.Circle(obj1.get_radius(), pos1)
	isIntersect = circ.is_intersect_line(gm.Line(pos2, pos2 + relVel))
	if not isIntersect:
		return tCol, nrmlCol, ptCol
	#Now we know that balls are definitely colliding.
	t = colDist / speed
	#Get the velocities along the direction of collision
	vCol1 = vel1.project(colDir)
	vCol2 = vel2.project(colDir)
	vOth1 = vel1 - vCol1
	vOth2 = vel2 - vCol2
	#Find the new velocities along the direction of collision
	m1 = obj1.get_mass()
	m2 = obj2.get_mass()
	#pdb.set_trace()
	vCol1New = (vCol1 * (m1 - m2) + 2 * m2 * vCol2) * (1.0 / (m1 + m2))
	vCol2New = (vCol2 * (m2 - m1) + 2 * m1 * vCol1) * (1.0 / (m1 + m2))
	vel1New  = vCol1New + vOth1
	vel2New  = vCol2New + vOth2
	obj1.set_after_collision_velocity(vel1New)			
	obj2.set_after_collision_velocity(vel2New)			
	#We dont require normal and point of collision.
	#pdb.set_trace()
	return t, None, None	
			

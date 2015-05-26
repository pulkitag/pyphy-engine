import geometry as gm

def test_line_intersection():
	#Intersect x, y axis
	l1 = gm.Line(gm.Point(-1,0), gm.Point(1,0))
	l2 = gm.Line(gm.Point(0,-1), gm.Point(0,1))
	print "GT:(0,0) Predict:", l1.get_intersection(l2)

	#45 lines
	l1 = gm.Line(gm.Point(0,0), gm.Point(1,1))
	l2 = gm.Line(gm.Point(1,0), gm.Point(0,1))
	print "GT:(0.5,0.5) Predict:", l1.get_intersection(l2)

	#Parallel lines
	l1 = gm.Line(gm.Point(0,0), gm.Point(1,1))
	l2 = gm.Line(gm.Point(1,3), gm.Point(5,7))
	print "GT:None Predict:", l1.get_intersection(l2)

	#Parallel lines
	l1 = gm.Line(gm.Point(0,-1), gm.Point(0,1))
	l2 = gm.Line(gm.Point(2,-1), gm.Point(2,1))	
	print "GT:None Predict:", l1.get_intersection(l2)


def test_line_ray_intersection():
	l1 = gm.Line(gm.Point(-5,5), gm.Point(5,5))
	l2 = gm.Line(gm.Point(-1,0), gm.Point(1,10))
	print "GT:(0,5) Predict:", l1.get_intersection_ray(l2)

	l1 = gm.Line(gm.Point(-5,5), gm.Point(5,5))
	l2 = gm.Line(gm.Point(-1,0), gm.Point(-1,3))
	print "GT:(-1,5) Predict:", l1.get_intersection_ray(l2)

	l1 = gm.Line(gm.Point(-5,5), gm.Point(5,5))
	l2 = gm.Line(gm.Point(-1,3), gm.Point(-1,0))
	print "GT:None Predict:", l1.get_intersection_ray(l2)


def test_line_bbox_intersection():
	bbox = gm.Bbox(gm.Point(1,2), gm.Point(1,1), gm.Point(2,1), gm.Point(2,2))
	l1   = gm.Line(gm.Point(0,0), gm.Point(1.5,1.9))
	l2   = gm.Line(gm.Point(0,0), gm.Point(1,10))
	l3   = gm.Line(gm.Point(0,1), gm.Point(5,1))	
	print "GT: True, Predict:",  bbox.is_intersect_line(l1)
	print "GT: False, Predict:", bbox.is_intersect_line(l2)
	print "GT: True, Predict:",  bbox.is_intersect_line(l3)
		
	pt1, s1 = bbox.get_intersection_with_line(l1)
	pt2, s2 = bbox.get_intersection_with_line(l2)
	pt3, s3 = bbox.get_intersection_with_line(l3)
	print pt1, s1
	print pt2, s2
	print pt3, s3	

def test_line_ray_bbox_intersection():
	bbox = gm.Bbox(gm.Point(1,2), gm.Point(1,1), gm.Point(2,1), gm.Point(2,2))
	l1   = gm.Line(gm.Point(0,0), gm.Point(1.5,1.9))
	l2   = gm.Line(gm.Point(0,0), gm.Point(1,10))
	l3   = gm.Line(gm.Point(0,1), gm.Point(5,1))	
	print "GT: True, Predict:",  bbox.is_intersect_line(l1)
	print "GT: False, Predict:", bbox.is_intersect_line(l2)
	print "GT: True, Predict:",  bbox.is_intersect_line(l3)

	l4   = gm.Line(gm.Point(1.5,0.5), gm.Point(1.5,-0.5))	
	l6   = gm.Line(gm.Point(1.5,-0.5), gm.Point(1.5, 0.5))	
	l5   = gm.Line(gm.Point(1.5,1.9), gm.Point(0,0))
	print "GT: False, Predict:",  bbox.is_intersect_line_ray(l4)
	print "GT: True, Predict:",   bbox.is_intersect_line(l4)
	print "GT: True, Predict:",   bbox.is_intersect_line_ray(l6)
	print "GT: True,  Predict:",  bbox.is_intersect_line_ray(l5)

def test_reflect():
	pt1 = gm.Point(-2,-1)
	rx  = (gm.Point(1,0)).reflect_normal(pt1)
	ry  = (gm.Point(0,1)).reflect_normal(pt1)
	print rx, ry

def test_point_along_line():
	pt1 = gm.Point(0,0)
	pt2 = gm.Point(1,1)
	l   = gm.Line(pt1, pt2)
	pt  = l.get_point_along_line(pt2, 3)
	print pt


def test_pseudo_tangent_contact():
	circle = gm.Circle(radius=20, center=gm.Point(0,0))
	#Parallel to x-axis
	pt1    = gm.Point(-1,25)
	pt2    = gm.Point(1,25)
	l1     = gm.Line(pt1, pt2)
	l2     = gm.Line(pt2, pt1)
	iPt1   = circle.get_contact_point_pseudo_tangent(l1)
	iPt2   = circle.get_contact_point_pseudo_tangent(l2)
	print "GT: (0,20), Predict: ", iPt1
	print "GT: (0,20), Predict: ", iPt2
	#Parallel to y-axis
	pt1    = gm.Point(-25,5)
	pt2    = gm.Point(-25,50)
	pt3    = gm.Point(30,5)
	pt4    = gm.Point(30,50)
	l1     = gm.Line(pt1, pt2)
	l2     = gm.Line(pt3, pt4)
	iPt1   = circle.get_contact_point_pseudo_tangent(l1)
	iPt2   = circle.get_contact_point_pseudo_tangent(l2)
	print "GT: (-20,0), Predict: ", iPt1
	print "GT: (20,0), Predict: ", iPt2
	#A diagonal
	pt1 = gm.Point(50,0)
	pt2 = gm.Point(0,50)
	l1     = gm.Line(pt1, pt2)
	iPt1   = circle.get_contact_point_pseudo_tangent(l1)
	print iPt1	
	

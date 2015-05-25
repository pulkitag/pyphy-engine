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


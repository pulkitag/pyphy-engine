import numpy as np
import copy
import pdb

class Point:
	def __init__(self, x=0, y=0):
		self.x_ = float(x)
		self.y_ = float(y)

	@classmethod
	def from_self(cls, other):
		self = cls(x=other.x(), y=other.y())
		return self

	def x(self):
		return self.x_

	def y(self):
		return self.y_

	def x_asint(self):
		return int(round(self.x_))

	def y_asint(self):
		return int(round(self.y_))

	def mag(self):
		return np.sqrt(self.x_ * self.x_ + self.y_ * self.y_)

	def scale(self, xScale, yScale=None):
		self.x_ = xScale * self.x_
		if yScale is None:
			self.y_ = xScale * self.y_
		else:
			self.y_ = yScale * self.y_
	
	def get_scaled_vector(self, scale):
		other = Point.from_self(self)
		other.make_unit_norm()
		other.scale(scale)
		return other

	#Make a point unit norm
	def make_unit_norm(self):
		mag = self.mag()
		if mag>0:
			self.scale(1.0/mag)

	#Dot product
	def dot(self, other):
		return self.x() * other.x() + self.y() * other.y()

	#Project self on some other point. 
	def project(self, other):
		otherUnit = Point.from_self(other)
		otherUnit.make_unit_norm()
		projMag  = self.dot(otherUnit)
		otherUnit.scale(projMag)
		return otherUnit	

	#Cosine angle between two vectors - assuming their origin to be zero. 
	def cosine(self, other):
		pt1 = Point.from_self(self)
		pt2 = Point.from_self(other)
		pt1.make_unit_norm()
		pt2.make_unit_norm()
		return pt1.dot(pt2)	

	#Reflect other with self as normal
	def reflect_normal(self, other):
		s = Point.from_self(self)
		s.make_unit_norm()
		prll = other.project(s)
		orth = other - prll
		#Reflect the orthogonal component
		prll.scale(-1)
		reflected = prll + orth
		return reflected

	def __add__(self, other):	
		p = Point()
		if isinstance(other, Point):
			p.x_ = self.x_ + other.x_
			p.y_ = self.y_ + other.y_
		else:
			uDir = Point.from_self(self)
			uDir.make_unit_norm()
			uDir.scale(other)
			p = self + uDir
		return p

	def __sub__(self, other):	
		p = Point()
		if isinstance(other, Point):
			p.x_ = self.x_ - other.x_
			p.y_ = self.y_ - other.y_
		else:
			uDir = Point.from_self(self)
			uDir.make_unit_norm()
			uDir.scale(other)
			p =  self - uDir
		return p

	def __mul__(self, scale):
		p = Point()
		p.x_ = self.x_ * scale
		p.y_ = self.y_ * scale
		return p

	__rmul__ = __mul__

	def __str__(self):
		return '(%.2f, %.2f)' % (self.x_, self.y_)

	#Does a point lie on quadrant 1 if the current point is the origin
	def is_quad1(self, pt):
		return pt.x() >= self.x_ and pt.y() >= self.y_

	#Quadrant-2		
	def is_quad2(self, pt):
		return pt.x() <= self.x_ and pt.y() >= self.y_

	#Quadrant-3
	def is_quad3(self, pt):
		return pt.x() <= self.x_ and pt.y() <= self.y_

	#Quadrant-4
	def is_quad4(self, pt):
		return pt.x() >= self.x_ and pt.y() <= self.y_

	#Distance
	def distance(self, pt, distType='L2'):
		if distType == 'L2':
			dist = (self - pt).mag()
		else:
			raise Exception('DistType: %s not recognized')	
		return dist


class Line:
	def __init__(self, pt1, pt2):
		#The line points from st_ to en_
		self.st_ = pt1
		self.en_ = pt2
		self.make_canonical()

	@classmethod
	def from_self(cls, other):
		self = cls(other.st(), other.en())
		return self

	def make_canonical(self):
		'''
			ax + by + c = 0
		'''
		self.a_ = float(-(self.en_.y() - self.st_.y()))
		self.b_ = float(self.en_.x() - self.st_.x())
		self.c_ = float(self.st_.x() * self.en_.y() - self.st_.y() * self.en_.x())
		aMag = np.abs(self.a_)
		if aMag > 0:
			self.a_ = self.a_ / aMag
			self.b_ = self.b_ / aMag
			self.c_ = self.c_ / aMag

	def a(self):
		return copy.deepcopy(self.a_)
	
	def b(self):
		return copy.deepcopy(self.b_)

	def c(self):
		return copy.deepcopy(self.c_)

	def st(self):
		return copy.deepcopy(self.st_)

	def mutable_st(self):
		return self.st_

	def en(self):
		return copy.deepcopy(self.en_)

	def mutable_en(self):
		return self.en_

	def get_direction(self):
		pt = self.en_ - self.st_
		pt.make_unit_norm()
		return pt

	#Returns the outward facing normal
	def get_normal(self):
		pt = Point(-self.a(), -self.b())
		pt.make_unit_norm()
		return pt	

	#Get the normal which points in the halfspace in
	#which point pt lies. 
	def get_normal_towards_point(self, pt):
		nrml = self.get_normal()
		ray   = Line(pt, pt + nrml)
		intPt = self.get_intersection_ray(ray)
		if intPt is None:
			return nrml
		else:
			nrml.scale(-1)
			return nrml 	

	def __str__(self):
		return "(%.2f, %.2f, %.2f)" % (self.a_, self.b_, self.c_) 

	#Returns the location of the point wrt a line
	def get_point_location(self, pt, tol=1e-6):
		'''
			returns: 1 is point is above the line (i.e. moving counter-clockwise from the line)
							-1 if the point is below
							 0 if on the line
		'''
		val = self.a_ * pt.x() + self.b_ * pt.y() + self.c_
		if val > tol:
			return 1
		elif val < -tol:
			return -1
		else:
			return 0 

	#Determines if the two points along the lie on the same line 
	#and if yes, what is their relative position. 
	def get_relative_location_points(self, pt1, pt2, tol=1e-06):
		'''
			returns: 0 is pt1 and pt2 donot lie on the line self
						 : 1 if pt2 is along self from pt1
						 :-1 if pt2 is the direction opposite of l1 from pt1
			Basically, we check
			x1 + \lamda l = x2
			=> \lamda l   = x2 - x1 (where \lamda is a scalar constant)
		'''
		ptDir = pt2 - pt1
		ptDir.make_unit_norm()
		lDir  = self.get_direction()
		cos   = ptDir.cosine(lDir)
		#print "ptDir: %f, lDir: %f, cos: %f" % (ptDir.mag(), lDir.mag(), cos)
		if (cos > 1 - tol) and (cos < 1 + tol):
			return 1
		elif (cos > -1 - tol) and (cos < -1 + tol):
			return -1
		else:
			return 0

	#Get a point along the line
	def get_point_along_line(self, pt, distance):
		lDir = self.get_direction()
		lDir.scale(distance)
		return pt + lDir

	#Intersection of two lines
	def get_intersection(self, l2):
		'''
			Point of intersection, y = (a2c1 - a1c2)/(a1b2 - a2b1)
			nr = a2c1 - a1c2
			dr = a1b2 - a2b1
		'''
		nr = l2.a() * self.c_ - self.a_ * l2.c()
		dr = self.a_ * l2.b() - l2.a() * self.b_
		#Parallel lines
		if dr == 0:
			return None
		else:
			y = nr / dr
			if self.a_ == 0:
				x = -(l2.c() + l2.b() * y) / l2.a() 
			else:
				x = -(self.c_ + self.b_ * y) / self.a_
		return Point(x, y)			 

	#Get intersection with a line ray
	def get_intersection_ray(self, l2):
		'''
			l2 is the ray
		'''
		pt = self.get_intersection(l2)
		if pt is not None:
			relLoc = l2.get_relative_location_points(l2.st(), pt)
			#print pt, relLoc
			if relLoc != 1:
				pt = None
		return pt

##
#Circle
class Circle:
	def __init__(self, radius=20, center=Point(0,0)):
		self.r_ = radius
		self.c_ = center		

	#Given a line l, find the line parallel to l
	#that is tangent to the circle and find the 
	#point of contact of this line with the circl
	def get_contact_point_pseudo_tangent(self, l):
		#Get the direction of the line from l to the center of the circle
		lToCDir = l.get_normal()	 
		#Scale to the radius
		lToCDir.scale(self.r_)
		#Get point of contact
		pt = self.c_ + lToCDir
		#Get equation of radius
		lr = Line(self.c_, pt)
		#Find if this line ray intersects the original line
		intersectPoint = l.get_intersection_ray(lr)
		if intersectPoint is not None:
			return pt
		else:
			#The point could be on the opposite direction
			lToCDir.scale(-1.0)
			pt = self.c_ + lToCDir
			lr = Line(self.c_, pt)
			intersectPoint = l.get_intersection_ray(lr)
			assert intersectPoint is not None
			return pt

##
# Note this not specifically a rectangular BBox. It can be in general be 
# of any shape. 
class Bbox:
	def __init__(self, lTop, lBot, rBot, rTop):
		'''
			lTop,..rTop: of Type Point
		'''
		self.vert_  = []
		self.vert_.append(lTop)
		self.vert_.append(lBot)
		self.vert_.append(rBot)
		self.vert_.append(rTop)
		self.N_ = len(self.vert_)
		self.lines_ = []
		self.make_lines()	
		
	@classmethod
	def from_list(cls, pts):
		pts  = copy.deepcopy(pts)
		assert len(pts)==4
		self = cls(pts[0], pts[1], pts[2], pts[3])
		return self

	#offset the bbox
	def move(self, offset):
		for i, vertex in enumerate(self.vert_):
			self.vert_[i] = vertex + offset
		self.make_lines()	
	
	#Make lines
	def make_lines(self):
		self.lines_ = []
		for i in range(self.N_):
			self.lines_.append(Line(self.vert_[i], self.vert_[np.mod(i+1,self.N_)]))

	def get_lines(self):
		return self.lines_

	#Determine if a point is inside the box or not
	def is_point_inside(self, pt):
		assert len(self.vert_)==4, 'Only works for rectangles'
		#If the lines are anticlockwise
		inside=True
		for l in self.lines_:
			inLine = l.get_point_location(pt)
			inside = inside and inLine >= 0
		#Clockwise lines
		clInside=True	
		for l in self.lines_:
			inLine = l.get_point_location(pt)
			clInside = clInside and inLine <= 0
		return (inside or clInside)

	#Determine if the bbox intersects with los(Line of Sight)
	def is_intersect_line(self, los):
		s = []
		isIntersect=True
		for i, v in enumerate(self.vert_):
			s.append(los.get_point_location(v))
			if i > 0:
				isIntersect = isIntersect and s[i]==s[i-1]
		isIntersect = not(isIntersect)
		return isIntersect

	#Determine if the bbox intersects with los that is a ray
	def is_intersect_line_ray(self, los):
		intPoint, dist = self.get_intersection_with_line_ray(los)
		if intPoint is not None:
			return True
		else:
			return False		
	
	#Find closest point
	def find_closest_interior_point(self, srcPt, pts, getIndex=False):
		'''
			from a list of Points (pts), find the point that is closest 
			to srcPt and is inside the Bbox
			if all points are outside None is returned. 
		'''
		intPoint = None
		dist     = np.inf
		idx      = None
		for i,pt in enumerate(pts):
			#No Intersection
			if pt is None:
				continue
			#Point of intersection is outside the bbox
			if not self.is_point_inside(pt):
				continue
			distTmp = srcPt.distance(pt)
			if distTmp  < dist:
				intPoint = pt
				dist     = distTmp
				idx      = i

		if getIndex:
			return intPoint, dist, idx
		else:
			return intPoint, dist

	#Point of intersection which is closest to the 
	#starting point of the line. 
	def get_intersection_with_line(self, l):
		'''
			Note this function considers l as a line and not a line segment
			If a line intersects, it is not necessary that a line segment will
			also intersect. 
		'''
		pts      = []
		boxLines = []
		for i,v in enumerate(self.vert_):
			boxLine = Line(v, self.vert_[np.mod(i+1, self.N_)])
			pts.append(l.get_intersection(boxLine))
			boxLines.append(boxLine)
		intPoint, dist, idx = self.find_closest_interior_point(l.st(), pts, getIndex=True)
		return intPoint, dist				

	#Find which line is intersected first by the ray
	def get_line_of_first_intersection_with_ray(self, l):
		pass
	
	#Point of intersection which is closest to the 
	#starting point of the line ray. 
	def get_intersection_with_line_ray(self, l):
		'''
			Note this function considers l as a line ray and not as line segment/line
		'''
		pts = []
		for i,v in enumerate(self.vert_):
			pts.append(Line(v, self.vert_[np.mod(i+1, self.N_)]).get_intersection_ray(l))
		return self.find_closest_interior_point(l.st(), pts)				
	
	#Get time of collision with another bounding box. 
	def get_toc_with_bbox(self, bbox, vel):
		'''
			self: is assumed to be stationary
			bbox: the other boundig bbox
			vel:  the velocity vector of bbox in frame of reference of self. 
		'''
		raise Exception('This function is not ready')
		pts = []
		pts.append(self.get_intersection_with_line(gm.Line(bbox.lTop_, bbox.lTop_ + vel))[0])
		pts.append(self.get_intersection_with_line(gm.Line(bbox.lBot_, bbox.lBot_ + vel))[0])
		pts.append(self.get_intersection_with_line(gm.Line(bbox.rBot_, bbox.rBot_ + vel))[0])
		pts.append(self.get_intersection_with_line(gm.Line(bbox.rTop_, bbox.rTop_ + vel))[0])
	

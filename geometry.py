import numpy as np

class Point:
	def __init__(self, x=0, y=0):
		self.x_ = x
		self.y_ = y

	@classmethod
	def from_self(cls, other):
		self = cls(x=other.x, y=other.y)
		return self

	def x(self):
		return self.x_

	def y(self):
		return self.y_

	def mag(self):
		return np.sqrt(self.x_ * self.x_ + self.y_ * self.y_)

	def scale(self, xScale, yScale=None):
		self.x_ = xScale * self.x_
		if yScale is None:
			self.y_ = xScale * self.y_
		else:
			self.y_ = yScale * self.y_

	def __add__(self, other):	
		p = Point()
		p.x_ = self.x_ + other.x_
		p.y_ = self.y_ + other.y_
		return p

	def __sub__(self, other):	
		p = Point()
		p.x_ = self.x_ - other.x_
		p.y_ = self.y_ - other.y_
		return p

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

	def make_canonical(self):
		'''
			ax + by + c = 0
		'''
		self.a_ = float(-(self.en_.y() - self.st_.y()))
		self.b_ = float(self.en_.x() - self.st_.x())
		self.c_ = float(self.st_.x() * self.en_.y() - self.st_.y() * self.en_.x())

	def a(self):
		return self.a_

	def b(self):
		return self.b_

	def c(self):
		return self.c_

	def st(self):
		return self.st_

	def en(self):
		return self.en_

	def __str__(self):
		return "(%.2f, %.2f, %.2f)" % (self.a_, self.b_, self.c_) 

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
		

class Bbox:
	def __init__(self, lTop, lBot, rBot, rTop):
		'''
			lTop,..rTop: of Type Point
		'''
		self.lTop_ = lTop
		self.lBot_ = lBot
		self.rBot_ = rBot
		self.rTop_ = rTop
		#self.l1_   = Line(lTop, lBot)
		#self.l2_   = Line(lBot, rBot)
		#self.l3_   = Line(rBot, rTop)
		#self.l4_   = Line(rTop, lTop)

	#offset the bbox
	def move(self, offset):
		self.lTop_ = self.lTop_ + offset
		self.rTop_ = self.rTop_ + offset
		self.lBot_ = self.lBot_ + offset
		self.rBot_ = self.rBot_ + offset
		
	#Determine if a point is inside the box or not
	def is_point_inside(self, pt):
		inside=True
		inside = inside and self.lTop_.is_quad4(pt)
		inside = inside and self.lBot_.is_quad1(pt)
		inside = inside and self.rBot_.is_quad2(pt)
		inside = inside and self.rTop_.is_quad3(pt)
		return inside

	#Determine if the bbox intersects with los
	def is_intersect_line(self, los):
		s1 = los.get_point_location(self.lTop_)
		s2 = los.get_point_location(self.lBot_)
		s3 = los.get_point_location(self.rBot_)
		s4 = los.get_point_location(self.rTop_)
		if s1==s2 and s2==s3 and s3==s4 and s1!=0:
			return False
		else:
			return True

	#Point of intersection which is closest to the 
	#starting point of the line.  
	def get_intersection_with_line(self, l):
		pts = []
		pts.append(l.get_intersection(Line(self.lTop_, self.lBot_)))
		pts.append(l.get_intersection(Line(self.lBot_, self.rBot_)))
		pts.append(l.get_intersection(Line(self.rBot_, self.rTop_)))
		pts.append(l.get_intersection(Line(self.rTop_, self.lTop_)))
		
		intPoint = None
		dist     = np.inf
		for i,pt in enumerate(pts):
			#No Intersection
			if pt is None:
				continue
			#Point of intersection is outside the bbox
			if not self.is_point_inside(pt):
				continue
			distTmp = l.st().distance(pt)
			if distTmp  < dist:
				intPoint = pt
				dist     = distTmp
		return intPoint, dist

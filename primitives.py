import numpy as np
import matplotlib.pyplot as plt
import collections as co
import cairo
import math
import pdb
import geometry as gm
import physics as phy
import copy
from collections import deque
import os
import scipy.io as sio
import scipy.misc as scm

class Color:
	def __init__(self, r, g, b, a=1.0):
		self.r = r
		self.g = g
		self.b = b
		self.a = a


class CairoData:
	def __init__(self, cr, im):
		self.cr = cr
		self.im = im

	
def get_ball_im(radius=40, fColor=Color(0.0, 0.0, 1.0), sThick=2, sColor=None):
	'''
		fColor: fill color
		sColor: stroke color
		sThick: stroke thickness
	'''
	sz      = 2*(radius + sThick)
	data    = np.zeros((sz, sz, 4), dtype=np.uint8)
	surface = cairo.ImageSurface.create_for_data(data, 
							cairo.FORMAT_ARGB32, sz, sz)
	cr      = cairo.Context(surface)
	#Create a transparent source
	cr.set_source_rgba(1.0, 1.0, 1.0, 0.0)
	cr.paint()
	#Create the border
	cx, cy = radius + sThick, radius + sThick
	cr.arc(cx, cy, radius, 0, 2*math.pi)
	cr.set_line_width(sThick)
	if sColor is not None:
		cr.set_source_rgba(sColor.b, sColor.g, sColor.r, sColor.a) 
	else:
		cr.set_source_rgba(0.0, 0.0, 0.0, 1.0) 
	cr.stroke()
	#Fill in the desired color
	cr.set_source_rgba(fColor.b, fColor.g, fColor.r, fColor.a)
	cr.arc(cx, cy, radius, 0, 2*math.pi)
	cr.fill()
	#cr.destroy()
	return cr, data


def get_rectangle_im(sz=gm.Point(4,100), fColor=Color(1.0, 0.0, 0.0)):
	data    = np.zeros((sz.y_asint(), sz.x_asint(), 4), dtype=np.uint8)
	surface = cairo.ImageSurface.create_for_data(data, 
							cairo.FORMAT_ARGB32, sz.x_asint(), sz.y_asint())
	cr      = cairo.Context(surface)
	#Create a transparent source
	cr.set_source_rgba(1.0, 1.0, 1.0, 0.0)
	cr.paint()
	# Make rectangle and fill in the desired color
	cr.set_source_rgba(fColor.b, fColor.g, fColor.r, fColor.a)
	cr.rectangle(0, 0, sz.x_asint(), sz.y_asint())
	cr.fill()
	#cr.destroy()
	return cr, data


def find_top_left(pts):
	xMin, yMin = np.inf, np.inf
	for p in pts:
		xMin = min(xMin, p.x())
		yMin = min(yMin, p.y())
	pt = gm.Point(xMin, yMin)
	return pt
	

def get_block_im(blockDir, fColor=Color(1.0, 0.0, 0.0), 
									sThick=2, bThick=30, sColor=None):
	'''
		blockDir: the the direction in which block needs to be created
	'''
	stPoint = gm.Point(0,0)
	enPoint = stPoint + blockDir
	pts = get_box_coords(stPoint, enPoint, wThick=bThick)
	pt1, pt2, pt3, pt4 = pts
	#Create the points for drawing the block.
	mnX   = min(pt1.x(), pt2.x(), pt3.x(), pt4.x())
	mnY   = min(pt1.y(), pt2.y(), pt3.y(), pt4.y())
	mnPt  = gm.Point(mnX, mnY)
	pt1, pt2  = pt1 - mnPt, pt2 - mnPt
	pt3, pt4  = pt3 - mnPt, pt4 - mnPt	
	print pt1, pt2, pt3, pt4

	if sColor is None:
		sColor = fColor
	xSz   = int(np.ceil(max(pt1.x(), pt2.x(), pt3.x(), pt4.x())))
	ySz   = int(np.ceil(max(pt1.y(), pt2.y(), pt3.y(), pt4.y())))
	data    = np.zeros((ySz, xSz, 4), dtype=np.uint8)
	surface = cairo.ImageSurface.create_for_data(data, 
							cairo.FORMAT_ARGB32, xSz, ySz)
	cr      = cairo.Context(surface)
	#Create a transparent source
	cr.set_source_rgba(1.0, 1.0, 1.0, 0.0)
	cr.paint()
	#Create the border/Mask
	cr.move_to(pt1.x(), pt1.y())
	cr.line_to(pt2.x(), pt2.y())
	cr.line_to(pt3.x(), pt3.y())
	cr.line_to(pt4.x(), pt4.y())
	cr.line_to(pt1.x(), pt1.y())
	cr.set_line_width(sThick)
	cr.set_source_rgba(sColor.b, sColor.g, sColor.r, sColor.a)
	cr.stroke()
	#Fill in the desired color
	cr.set_source_rgba(fColor.b, fColor.g, fColor.r, fColor.a)
	cr.move_to(pt1.x(), pt1.y())
	cr.line_to(pt2.x(), pt2.y())
	cr.line_to(pt3.x(), pt3.y())
	cr.line_to(pt4.x(), pt4.y())
	cr.line_to(pt1.x(), pt1.y())
	cr.fill()
	return cr, data


def get_box_coords(stPoint, enPoint, wThick=30):
	'''
		stPoint: bottom-left point
		enPoint: The direction along which thickness needs to expanded. 
		wThick: thickness of the wall
	'''
	line = gm.Line(stPoint, enPoint)
	pDiff = enPoint - stPoint
	dist  = pDiff.mag()
	nrml  = line.get_normal()
	lDir  = line.get_direction()
	#pt1   = stPoint + (bThick/2.0) * nrml 
	pt1   = stPoint 
	pt2   = pt1 + dist * lDir
	pt3   = pt2 - (wThick * nrml)
	pt4   = pt3 - (dist * lDir)
	pts   = [pt1, pt2, pt3, pt4]
	print pt1, pt2, pt3, pt4
	return pts	


#Create a cage
def create_cage(pts, wThick=30):
	'''
		pts: a list of points
	'''
	N        = len(pts)
	walls    = []
	for i,pt in enumerate(pts):
		stPoint = pts[i]
		enPoint = pts[np.mod(i+1,N)]
		walls.append(GenericWall(stPoint, enPoint, wThick=wThick))
	return walls		

##
#Wall def just defines the physical properties. 
class WallDef:
	def __init__(self,  sz=gm.Point(4, 100), 
							 fColor=Color(1.0, 0.0, 0.0), name=None):
		self.sz        = sz
		self.fColor    = fColor
		self.name      = name


class GenericWall:
	def __init__(self, stPoint, enPoint, fColor=Color(1.0, 0.0, 0.0),
							 name=None, wThick=4):
		self.pts_ = get_box_coords(stPoint, enPoint, wThick=wThick)	
		self.th_  = wThick
		self.pos_ = find_top_left(self.pts_) #This would be the top-left point
		self.make_data(stPoint, enPoint)
		self.bbox_ = gm.Bbox.from_list(self.pts_)

	#Make the image data
	def make_data(self, stPoint, enPoint):
		cr, im = get_block_im(enPoint - stPoint, bThick=self.th_)	
		self.data_  = CairoData(cr, im)	
		self.imSzY_, self.imSzX_ ,_  = im.shape

	#Imprint the data on the canvas	
	def imprint(self, cr, xSz, ySz):
		'''
			xSz, ySz: Size of the canvas on which imprint has to be made. 
		'''
		#Create Source
		y, x  = self.pos_.y_asint(), self.pos_.x_asint()		
		srcIm = np.zeros((ySz, xSz, 4), dtype=np.uint8)
		print "pos: (%f, %f), sz:(%f, %f)" % (x, y, self.imSzX_, self.imSzY_)
		srcIm[y : y + self.imSzY_, x : x + self.imSzX_,:] = self.data_.im[:] 
		surface = cairo.ImageSurface.create_for_data(srcIm, 
								cairo.FORMAT_ARGB32, xSz, ySz)
		cr.set_source_surface(surface)	
		#Create Mask
		pt1, pt2, pt3, pt4 = self.pts_	
		cr.move_to(pt1.x(), pt1.y())
		cr.line_to(pt2.x(), pt2.y())
		cr.line_to(pt3.x(), pt3.y())
		cr.line_to(pt4.x(), pt4.y())
		cr.line_to(pt1.x(), pt1.y())
		#Fill source into the mask
		cr.fill()


##
# Defines the physical properties along with the location etc of the object.
# This is a rectangular wall 
class Wall:
	def __init__(self, initPos=gm.Point(0, 0), sz=gm.Point(4, 100), 
							 fColor=Color(1.0, 0.0, 0.0), name=None):
		'''
			initPos: Upper left corner
		'''
		self.sz_     = sz
		self.pos_    = initPos
		self.fColor_ = fColor
		self.name_   = name
		self.make_coordinates()
		self.make_data()

	@classmethod
	def from_def(cls, wallDef, name, initPos):
		self       = cls(sz=wallDef.sz, fColor=wallDef.fColor, 
										 initPos=initPos, name=name)
		return self

	#Make cairo data
	def make_data(self):
		cr, im = get_rectangle_im(sz=self.sz_, fColor=self.fColor_)
		self.data_      = CairoData(cr, im)	

	#Make the coordinates of the four corners
	def make_coordinates(self):
		self.lTop_ = self.pos_
		self.lBot_ = self.pos_ + gm.Point(0, self.sz_.y())
		self.rTop_ = self.pos_ + gm.Point(self.sz_.x(), 0)
		self.rBot_ = self.pos_ + gm.Point(self.sz_.x(), self.sz_.y())
		self.l1_   = gm.Line(self.lTop_, self.lBot_) #Left line
		self.l2_   = gm.Line(self.lBot_, self.rBot_)
		self.l3_   = gm.Line(self.rBot_, self.rTop_)
		self.l4_   = gm.Line(self.rTop_, self.lTop_)
		self.bbox_ = gm.Bbox(self.lTop_, self.lBot_, self.rBot_, self.rTop_)

	#Imprint the wall
	def imprint(self, cr, xSz, ySz):
		'''
			xSz, ySz: Size of the canvas on which imprint has to be made. 
		'''
		y, x  = self.pos_.y(), self.pos_.x()		
		srcIm = np.zeros((ySz, xSz, 4), dtype=np.uint8)
		srcIm[y : y + self.sz_.y(), x : x + self.sz_.x(),:] = self.data_.im[:] 
		surface = cairo.ImageSurface.create_for_data(srcIm, 
								cairo.FORMAT_ARGB32, xSz, ySz)
		cr.set_source_surface(surface)		
		cr.rectangle(x, y, self.sz_.x(), self.sz_.y())
		cr.fill()
		#print "Wall- x: %d, y: %d, szX: %d, szY: %d" % (x, y, self.sz_.x(), self.sz_.y())

	#Gives the normal that are needed to solve for collision. 
	def get_collision_normal(self, pt):
		'''
			The wall has 4 faces and a normal associated with each of these faces.
			we would basically determine where the point lies and then generate
			collisions based on that. 
		'''	
		#If the particle is to the left. 
		if ((self.l1_.get_point_location(pt) == -1 and self.l3_.get_point_location(pt)==1) or
				 self.l1_.get_point_location(pt) == 0):
			return self.l1_.get_normal()
		
		#If the particle is on the right. 	
		if ((self.l1_.get_point_location(pt) == 1 and self.l3_.get_point_location(pt) == -1) or
				 self.l3_.get_point_location(pt) == 0):
			return self.l3_.get_normal()

		#If the particle is on the top
		if ((self.l2_.get_point_location(pt) == 1 and self.l4_.get_point_location(pt) == -1) or
				 self.l4_.get_point_location(pt) == 0):
			return self.l4_.get_normal()

		#If the particle is on the bottom
		if ((self.l2_.get_point_location(pt) == -1 and self.l4_.get_point_location(pt) == 1) or
				 self.l2_.get_point_location(pt) == 0):
			return self.l2_.get_normal()

	#Check collision with a velocity vector. 
	def check_collision(self, headingDir):
		'''
			headingDir: The headingDirection with which some other object is moving
			Check is this object will collide the wall or not. 
		'''
		intersectionPoint, dist = self.bbox_.get_intersection_with_line_ray(headingDir)
		return intersectionPoint, dist

	def name(self):
		return self.name_


##
#Ball def just defines the physical properties. 
class BallDef:
	def __init__(self, radius=20, sThick=2, 
							 sColor=Color(0.0, 0.0, 0.0), fColor=Color(1.0, 0.0, 0.0),
							 name=None, density=1.0):
		self.radius    = radius
		self.sThick    = sThick
		self.sColor    = sColor
		self.fColor    = fColor
		self.name      = name
		self.density   = density
		
##
# Defines the physical properties along with the location etc of the object. 
class Ball:
	def __init__(self, radius=20, sThick=2, 
							 sColor=Color(0.0, 0.0, 0.0), fColor=Color(1.0, 0.0, 0.0),
							 name=None, initPos=gm.Point(0,0), initVel=gm.Point(0,0),
							 density=1.0):
		self.radius_    = radius
		self.sThick_    = sThick
		self.sColor_    = sColor
		self.fColor_    = fColor
		self.name_      = name
		self.pos_       = initPos
		self.tCol_      = 0
		self.vel_       = initVel
		self.density_   = density
		self.mass_      = (4/3.0) * np.pi * np.power(self.radius_/10.0,3) * density
		self.make_data()

	@classmethod
	def from_def(cls, ballDef, name, initPos, initVel=gm.Point(0,0)):
		self = cls(radius=ballDef.radius, sThick=ballDef.sThick, 
							 sColor=ballDef.sColor, fColor=ballDef.fColor)
		self.name_ = name
		self.pos_  = initPos
		self.vel_  = initVel
		return self

	def set_name(self, name):
		self.name_ = name

	#Make cairo data
	def make_data(self):
		cr, im      = get_ball_im(radius=self.radius_, fColor=self.fColor_,
														  sThick=self.sThick_, sColor=self.sColor_)
		self.data_  = CairoData(cr, im)	
		self.ySz_, self.xSz_   = im.shape[0], im.shape[1]
		self.yOff_, self.xOff_ = np.floor(self.ySz_ / 2), np.floor(self.xSz_ / 2)

	#Imprint the ball
	def imprint(self, cr, xSz, ySz):
		#Get the position of bottom left corner.
		y, x  = self.pos_.y_asint() - self.yOff_, self.pos_.x_asint() - self.xOff_		
		srcIm = np.zeros((ySz, xSz, 4), dtype=np.uint8)
		srcIm[y:y+self.ySz_, x:x+self.xSz_,:] = self.data_.im[:] 
		surface = cairo.ImageSurface.create_for_data(srcIm, 
								cairo.FORMAT_ARGB32, xSz,ySz)
		cr.set_source_surface(surface)		
		cr.rectangle(x, y, self.xSz_, self.ySz_)
		cr.fill()

	def name(self):
		return self.name_

	def get_position(self):
		return copy.deepcopy(self.pos_)

	def get_mutable_position(self):
		return self.pos_	
	
	def get_velocity(self):
		return copy.deepcopy(self.vel_)

	def get_mutable_velocity(self):
		return self.vel_	

	def set_position(self, pos):
		self.pos_ = copy.deepcopy(pos)

	def set_velocity(self, vel):
		self.vel_ = copy.deepcopy(vel)

	def get_mass(self):
		return self.mass_

	def set_mass(self, mass):
		self.mass_ = mass

	#The direction in which the ball is heading with the
	#center of the ball as starting point of the line 
	def get_heading_direction_line(self):
		headDir = self.vel_.get_scaled_vector(self.radius_)
		#st = self.pos_ + headDir
		st = self.pos_
		en = st + self.vel_
		return gm.Line(st, en)

	#Get point of collision on the point given the point the ball is going to collide on. 
	def get_point_of_contact(self, pt):
		'''
			pt: point to which the ball is going to collide
		'''
		headDir = self.get_heading_direction_line()
		assert headDir.get_point_location(pt)==0, 'The pt of collision and heading direction donot match'
		#Get the point on the ball which will collide.
		ptBall  = headDir.get_point_along_line(self.pos_, self.radius_)
		return ptBall
	

	
class Dynamics:
	def __init__(self, world, g=0, deltaT=0.01):
		'''
			g: gravity, since the (0,0) is top left, gravity will be positive. 
		'''
		self.world_  = world
		self.g_      = gm.Point(0, g)
		self.deltaT_ = deltaT
		#Record which objects have been stepped and which have not been.
		self.isStep_ = co.OrderedDict()
		#Record time to collide and object of collision
		self.tCol_   = co.OrderedDict()
		self.objCol_ = co.OrderedDict()
		for name in self.get_dynamic_object_names():
			self.isStep_[name] = False
			self.tCol_[name] = 0
			self.objCol_[name] = None

	#Set gravity
	def set_g(self, g):
		self.g_ = g

	#Get names of objects
	def get_dynamic_object_names(self):
		return self.world_.get_dynamic_object_names()

	def move_object(self, obj, deltaT):
		pos = obj.get_mutable_position()
		vel = obj.get_mutable_velocity()
		#Update position: s = ut + 0.5at^2
		pos = pos + deltaT * vel + (0.5 * deltaT * deltaT) * self.g_
		#Update velocity: v = u + at
		vel = vel + deltaT * self.g_
		obj.set_position(pos)
		obj.set_velocity(vel)
		#print pos, vel

	#1 time step
	def step_object(self, obj, name):
		if self.isStep_[name]:
			return
		if self.tCol_[name] < self.deltaT_:
			self.resolve_collision(obj, name)
			return
		oldVel = obj.get_velocity()
		self.move_object(obj, self.deltaT_)
		self.tCol_[name]   = self.tCol_[name] - self.deltaT_
		#print self.tCol_[name]
		#If a sationary object has been set to motion, then perform resolve_collision
		if oldVel.mag() == 0 and (obj.get_velocity()).mag() > 0:
			self.resolve_collision(obj, name) 
		self.isStep_[name] = True

	#Apply force on an object
	def apply_force(self, objName, force, forceT=None):
		'''
			forceT: amount of time for which force is applied. 
		'''
		if forceT is None:
			forceT = self.deltaT_
		obj  = self.get_object(objName)
		mass = obj.get_mass()
		assert mass > 0, "Mass has to be a positive number"
		a      = gm.Point.from_self(force)
		a.scale(1.0/mass) 
		deltaV = 0.5 * forceT * forceT * a
		vel    = obj.get_velocity()
		vel    = vel + deltaV
		obj.set_velocity(vel)	 

	#Step the entire world
	def step(self):
		#Reset step states
		for name in self.get_dynamic_object_names():
			self.isStep_[name] = False
		#Step every object
		for name in self.get_dynamic_object_names():
			self.step_object(self.world_.get_object(name), name)
		 
	#Resolve the collisions
	def resolve_collision(self, obj, name, deltaT=None):
		if deltaT is None:
			deltaT = self.deltaT_
		pos = obj.get_position()
		vel = obj.get_velocity()
		#Stationary object
		if vel.mag() == 0:
			self.tCol_[name] = np.inf
			self.step_object(obj, name)
			return	
		#Stationary object just set into motion
		if self.tCol_[name]==np.inf and vel.mag() > 0:
			self.time_to_collide_all(obj, name)
			return

		#This can happen when there is intiial velocity provided to the object. 
		if self.tCol_[name]==0 and self.objCol_[name] is None:
			self.time_to_collide_all(obj, name)
			return

		#Resolve an actual collision. 
		obj2 = self.objCol_[name]
		if isinstance(obj, Ball) and isinstance(obj2, Wall):
			#Move the object to the position of collision
			assert self.tCol_[name] <= deltaT
			extraT = deltaT - self.tCol_[name]	
			self.move_object(obj, self.tCol_[name])
			self.tCol_[name] = 0

			headingDir           = obj.get_heading_direction_line()
			intersectPoint, dist = obj2.check_collision(headingDir)
			if intersectPoint is None:
				pdb.set_trace()
			assert intersectPoint is not None, 'intersection point cannot be None'
			ptBall = obj.get_point_of_contact(intersectPoint)
			toc    = phy.time_from_pt1_pt2(ptBall, intersectPoint, obj.get_velocity())
			assert toc < deltaT, 'Ball should be colliding now, it is not'		
			#print toc
			#Get normals from the wall
			nrml = obj2.get_collision_normal(intersectPoint)
			#Compute the new velocity
			vel  = nrml.reflect_normal(obj.get_velocity()) 
			#Set the new velocity
			obj.set_velocity(vel)
			#Check for iminent collisions
			self.time_to_collide_all(obj, name)
			if self.tCol_[name] > extraT:
				self.move_object(obj, extraT)
			else:
				self.resolve_collision(obj, name, deltaT=extraT)			
			#Move the ball for extraT
			#self.move_object(obj, extraT)
			#self.tCol_[name] = np.inf
			#self.objCol_[name] = None
			#self.resolve_collision(obj, name)
		else:
			print "Type obj1:", type(obj)
			print "Type obj2:", type(obj2)
			raise Exception('Collision type not recognized') 	


	def time_to_collide(self, obj1, obj2):
		'''
			obj1, obj2: detection collision of object 1 with object 2
		'''
		if isinstance(obj1, Ball) and isinstance(obj2, Wall):
			headingDir          = obj1.get_heading_direction_line()
			intersectPoint, dist = obj2.check_collision(headingDir)
			#pdb.set_trace()
			if intersectPoint is not None:
				ptBall = obj1.get_point_of_contact(intersectPoint)
				toc    = phy.time_from_pt1_pt2(ptBall, intersectPoint, obj1.get_velocity())
				#pdb.set_trace()
			else:
				toc = np.inf  	
		else:
			raise Exception('Collision type not recognized')
		return toc	
	
	#Get time to collision
	def time_to_collide_all(self, obj, name):
		#Reset toc
		self.tCol_[name]    = np.inf
		self.objCol_[name]  = None
		allNames = self.world_.get_object_names()
		for an in allNames:
			if an == name:
				continue
			toc = self.time_to_collide(obj, self.get_object(an))
			#print toc
			if toc < self.tCol_[name]:
				self.tCol_[name]   = toc
				self.objCol_[name] = self.get_object(an)
		#print self.tCol_[name]	
	
	#Get the image
	def generate_image(self):
		return self.world_.generate_image()
								
	#Get the object
	def get_object(self, name):
		return self.world_.get_object(name)

	#Get the dynamic object
	def get_dynamic_object_names(self):
		return self.world_.get_dynamic_object_names()

	#Get object position
	def get_object_position(self, name):
		return self.world_.get_object_position(name)


#This is useful if one needs to have a lookahead. 
class DynamicsHorizon:
	def __init__(self, model, lookAhead=20):
		self.N_     = lookAhead
		self.model_ = model
		self.names_ = model.get_dynamic_object_names()
		#Stores the positions of the dynamic objects 
		self.pos_ = co.OrderedDict()
		for name in self.names_:
			self.pos_[name] = deque()
		#Store the images
		self.im_   = deque()
		#Store the temporary positions and image
		self.tmpPos_ = co.OrderedDict()
		self.tmpIm_  = None
		#Output matrix for positions
		self.outMat_ = np.zeros((len(self.names_), self.N_ * 2)).astype(np.float32)
		for i in range(lookAhead):
			im = self.step()
			self.im_.append(im)
			for name in self.names_:
				self.pos_[name].append(self.tmpPos_[name])

	#
	def step(self):
		self.model_.step()
		for name in self.names_:
			self.tmpPos_[name] = self.model_.get_object_position(name)
		self.tmpIm_ = self.model_.generate_image()	
				
	#
	def get_data(self):
		self.step()
		#Get the image
		im = self.im_.popleft()
		self.im_.append(self.tmpIm_)	
		#Get the velocities on the ball on the horizon
		for i, name in enumerate(self.names_):
			self.pos_[name].append(self.tmpPos_[name])
			for n in range(self.N_):
				vel = self.pos_[name][i+1] - self.pos_[name][i]
				self.outMat_[i, n*2] = np.float32(vel.x())
				self.outMat_[i, n*2 + 1] = np.float32(vel.y())
			_ = self.pos_[name].popleft() 
		return copy.deepcopy(im), copy.deepcopy(self.outMat_)


#The world
class World:
	def __init__(self, xSz=200, ySz=200, deltaT=0.1):
		'''
			xSz, ySz: Size of the cavas that defines the world.
			deltaT: the stepping time used in the simulation. 
		'''
		#Static Objects
		self.static_  = co.OrderedDict()
		#Dynamic Objects
		self.dynamic_ = co.OrderedDict()
		#gm.Pointers to all objects
		self.objects_ = co.OrderedDict()
		#Count of objects
		self.count_   = co.OrderedDict()
		self.init_count()
		#Form the base canvas
		self.xSz_     = xSz
		self.ySz_     = ySz 
		cr, im        = get_rectangle_im(sz=gm.Point(xSz,ySz), fColor=Color(1.0,1.0,1.0))
		self.baseCanvas_ = CairoData(cr, im)

	#Contains the names of primitives that the world
	# can have.
	def init_count(self):
		self.count_['wall'] = 0
		self.count_['ball'] = 0

	#Names of dynamic objects
	def get_dynamic_object_names(self):
		return self.dynamic_.keys()

	#Get object names
	def get_object_names(self):
		return self.objects_.keys()

	##
	def get_object(self, name):
		assert name in self.objects_.keys(), 'Object: %s not found' % name
		return self.objects_[name]

	##
	def get_object_name_type(self, objDef, initPos=None):
		if isinstance(objDef, WallDef):
			name    = 'wall-%d' % self.count_['wall']
			obj     = Wall.from_def(objDef, name, initPos)
			objType = 'static' 
			self.count_['wall'] += 1
		elif isinstance(objDef, GenericWall):
			name    = 'wall-%d' % self.count_['wall']
			obj     = objDef
			objType = 'static' 
			self.count_['wall'] += 1
		elif isinstance(objDef, BallDef):
			name    = 'ball-%d' % self.count_['ball']					
			obj     = Ball.from_def(objDef, name, initPos)
			objType = 'dynamic'
			self.count_['ball'] += 1
		else:
			raise Exception('Unrecognized object type')
		
		return obj, name, objType

	##
	def add_object(self, objDef, initPos=gm.Point(0,0)):
		'''
			objDef : The definition of the object that needs to be added.
			initPos: The initial position of the object in the normalized coords.
		'''
		obj, name, objType = self.get_object_name_type(objDef, initPos)
		if name in self.objects_.keys():
			raise Exception('Object with name %s already exists' % name)
		self.objects_[name] = obj
		if objType == 'static':
			self.static_[name]  = obj
		else:
			self.dynamic_[name] = obj		

	#
	def get_object_position(self, objName):
		return self.objects_[objName].get_position()
	
	#
	def set_object_position(self, objName, newPos):
		assert objName in self.dynamic_.keys()
		self.dynamic_[objName].set_position(newPos)

	def increment_object_position(self, objName, deltaPos):
		assert objName in self.dynamic_.keys()
		self.dynamic_[objName].set_position(self.dynamic_[objName].get_position() + deltaPos)

	def generate_image(self):
		data    = np.zeros((self.ySz_, self.xSz_, 4), dtype=np.uint8)
		data[:] = self.baseCanvas_.im[:]
		surface = cairo.ImageSurface.create_for_data(data, 
								cairo.FORMAT_ARGB32, self.xSz_, self.ySz_)
		cr      = cairo.Context(surface)
		for key, obj in self.objects_.iteritems():
			obj.imprint(cr, self.xSz_, self.ySz_)
		return data

import numpy as np
import matplotlib.pyplot as plt
import collections as co
import cairo
import math
import pdb

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


class Point:
	def __init__(self, x, y):
		self.x_ = x
		self.y_ = y

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


def get_rectangle_im(sz=Point(4,100), fColor=Color(1.0, 0.0, 0.0)):
	data    = np.zeros((sz.y(), sz.x(), 4), dtype=np.uint8)
	surface = cairo.ImageSurface.create_for_data(data, 
							cairo.FORMAT_ARGB32, sz.y(), sz.x())
	cr      = cairo.Context(surface)
	#Create a transparent source
	cr.set_source_rgba(1.0, 1.0, 1.0, 0.0)
	cr.paint()
	# Make rectangle and fill in the desired color
	cr.set_source_rgba(fColor.b, fColor.g, fColor.r, fColor.a)
	cr.rectangle(0, 0, sz.y(), sz.x())
	cr.fill()
	#cr.destroy()
	return cr, data


##
#Wall def just defines the physical properties. 
class WallDef:
	def __init__(self,  sz=Point(4, 100), 
							 fColor=Color(1.0, 0.0, 0.0), name=None):
		self.sz        = sz
		self.fColor    = fColor
		self.name      = name
		
##
# Defines the physical properties along with the location etc of the object. 
class Wall:
	def __init__(self, initPos=Point(0, 0), sz=Point(4, 100), 
							 fColor=Color(1.0, 0.0, 0.0), name=None):
		self.sz_     = sz
		self.pos_    = initPos
		self.fColor_ = fColor
		self.name_   = name
		self.make_data()

	@classmethod
	def from_def(cls, wallDef, name, initPos):
		self       = cls(sz=wallDef.sz, fColor=wallDef.fColor)
		self.name_ = name
		self.pos_  = initPos
		return self

	#Make cairo data
	def make_data(self):
		cr, im = get_rectangle_im(sz=self.sz_, fColor=self.fColor_)
		self.data_      = CairoData(cr, im)	

	def name(self):
		return self.name_


##
#Ball def just defines the physical properties. 
class BallDef:
	def __init__(self, radius=20, sThick=2, 
							 sColor=Color(0.0, 0.0, 0.0), fColor=Color(1.0, 0.0, 0.0),
							 name=None):
		self.radius    = radius
		self.sThick    = sThick
		self.sColor    = sColor
		self.fColor    = fColor
		self.name      = name

		
##
# Defines the physical properties along with the location etc of the object. 
class Ball:
	def __init__(self, radius=20, sThick=2, 
							 sColor=Color(0.0, 0.0, 0.0), fColor=Color(1.0, 0.0, 0.0),
							 name=None, initPos=Point(0,0)):
		self.radius_    = radius
		self.sThick_    = sThick
		self.sColor_    = sColor
		self.fColor_    = fColor
		self.name_      = name
		self.pos_       = initPos
		self.make_data()

	@classmethod
	def from_def(cls, ballDef, name, initPos):
		self = cls(radius=ballDef.radius, sThick=ballDef.sThick, 
							 sColor=ballDef.sColor, fColor=ballDef.fColor)
		self.name_ = name
		self.pos_  = initPos
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
		y, x  = self.pos_.y() - self.yOff_, self.pos_.x() - self.xOff_		
		srcIm = np.zeros((xSz, ySz, 4), dType=np.uint8)
		srcIm[y:y+self.ySz_, x:x+self.xSz_,:] = self.data_.im[:,:,:] 

	def name(self):
		return self.name_


#The world
class World:
	def __init__(self, xSz=200, ySz=200):
		#Static Objects
		self.static_  = co.OrderedDict()
		#Dynamic Objects
		self.dynamic_ = co.OrderedDict()
		#Pointers to all objects
		self.objects_ = co.OrderedDict()
		#Count of objects
		self.count_   = co.OrderedDict()
		self.init_count()
		#Form the base canvas
		self.xSz_     = xSz
		self.ySz_     = ySz 
		cr, im        = get_rectangle_im(sz=Point(xSz,ySz), fColor=Color(1.0,1.0,1.0))
		self.baseCanvas_ = CairoData(cr, im)

	#Contains the names of primitives that the world
	# can have.
	def init_count(self):
		self.count_['wall'] = 0
		self.count_['ball'] = 0

	##
	#
	def get_object_name_type(self, objDef, initPos):
		if isinstance(objDef, WallDef):
			name    = 'wall-%d' % self.count_['wall']
			obj     = Wall.from_def(objDef, name, initPos)
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
	#
	def add_object(self, objDef, initPos=(0,0)):
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
	

	def generate_image(self):
		data    = np.zeros((self.ySz_, self.xSz_, 4), dtype=np.uint8)
		data[:] = self.baseCanvas_.im[:]
		surface = cairo.ImageSurface.create_for_data(data, 
								cairo.FORMAT_ARGB32, self.ySz_, self.xSz_)
		cr      = cairo.Context(surface)
		

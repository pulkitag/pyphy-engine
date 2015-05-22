import cairo
import numpy as np
import primitives as pm
import matplotlib.pyplot as plt

##
#Try to paste a ball at a certain location
def paste_ball():
	bDef  = pm.BallDef()
	ball1 = pm.Ball.from_def(bDef, 'ball1', pm.Point(70,50))

	#Canvas Sz
	xCsz, yCsz = 640, 480
	#Create the base to paste it on
	data    = np.zeros((yCsz, xCsz, 4), dtype=np.uint8)
	surface = cairo.ImageSurface.create_for_data(data, 
							cairo.FORMAT_ARGB32, xCsz, yCsz)
	cr      = cairo.Context(surface)
	cr.set_source_rgba(0.5, 0.5, 1.0, 1.0)
	cr.paint()

	#Create imSurface
	shp = ball1.data_.im.shape
	y, x  = ball1.pos_.y() - ball1.yOff_, ball1.pos_.x() - ball1.xOff_
	imDat    = np.zeros((yCsz, xCsz, 4), dtype=np.uint8)
	imDat[y:y+shp[0],x:x+shp[0],:] = ball1.data_.im[:] 
	surface = cairo.ImageSurface.create_for_data(imDat, 
							cairo.FORMAT_ARGB32, xCsz, yCsz)
	cr.set_source_surface(surface)		
	#cr.set_source_rgb(1.0,0.0,0.0)		
	cr.rectangle(x, y, shp[0], shp[1])
	cr.fill()
	print y,x,shp[0],shp[1]
	return data	


def create_ball_world():
	world = pm.World(xSz=640, ySz=480)
	bDef  = pm.BallDef()
	bDef2 = pm.BallDef(fColor=pm.Color(0.0,0.0,1.0))
	wallHorDef = pm.WallDef(sz=pm.Point(600, 4))
	wallVerDef = pm.WallDef(sz=pm.Point(4, 450))
	#Horizontal Wall
	world.add_object(wallHorDef, initPos=pm.Point(20,20))
	world.add_object(wallHorDef, initPos=pm.Point(20,470))
	world.add_object(wallVerDef, initPos=pm.Point(20,20))
	world.add_object(wallVerDef, initPos=pm.Point(616,20))
	world.add_object(bDef, initPos=pm.Point(200,200))
	world.add_object(bDef2, initPos=pm.Point(400,400))
	im = world.generate_image()	
	return im, world	

def create_ball_world_gray():
	wThick = 30
	world = pm.World(xSz=640, ySz=480)
	bDef  = pm.BallDef(fColor=pm.Color(0.5,0.5,0.5))
	bDef2 = pm.BallDef(fColor=pm.Color(0.5,0.5,0.5))

	xLength, yLength = 550, 400
	wallHorDef = pm.WallDef(sz=pm.Point(xLength, wThick), fColor=pm.Color(0.5,0.5,0.5))
	wallVerDef = pm.WallDef(sz=pm.Point(wThick, yLength), fColor=pm.Color(0.5,0.5,0.5))

	xLeft, yTop = 30, 30
	#Horizontal Wall
	world.add_object(wallHorDef, initPos=pm.Point(xLeft, yTop))
	world.add_object(wallHorDef, initPos=pm.Point(xLeft, yTop + yLength))
	world.add_object(wallVerDef, initPos=pm.Point(xLeft, yTop))
	world.add_object(wallVerDef, initPos=pm.Point(xLeft + xLength -wThick, yTop))
	world.add_object(bDef, initPos=pm.Point(200,200))
	world.add_object(bDef2, initPos=pm.Point(400,400))
	im = world.generate_image()	
	return im, world	



def ball_world_simulation():
	im, world = create_ball_world_gray()
	plt.ion()
	for i in range(10):
		world.increment_object_position('ball-1', pm.Point(5,5))
		im = world.generate_image()
		plt.imshow(im)
		raw_input()	

import cairo
import numpy as np
import primitives as pm
import geometry as gm
import matplotlib.pyplot as plt
from matplotlib import animation
from collections import deque
import scipy.misc as scm

tmpDir = '/data1/pulkitag/tmp/projPhysics/'
imName = tmpDir + 'im%d.jpg'

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


def create_world_diamond():
	world = pm.World(xSz=640, ySz=480)
	pt1   = gm.Point(50, 240)
	pt2   = gm.Point(300, 40)
	pt3   = gm.Point(550, 240)
	pt4   = gm.Point(300, 440)
	walls = pm.create_cage([pt1, pt2, pt3, pt4], wThick=20)
	for w in walls:
		world.add_object(w)

	bDef  = pm.BallDef(fColor=pm.Color(0.5,0.5,0.5))
	world.add_object(bDef, initPos=gm.Point(200,200))
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


def ball_world_simulation_nophysics():
	im, world = create_ball_world_gray()
	plt.ion()
	for i in range(10):
		world.increment_object_position('ball-1', pm.Point(5,5))
		im = world.generate_image()
		plt.imshow(im)
		raw_input()	


def create_single_ball_world_gray():
	wThick = 30
	world = pm.World(xSz=640, ySz=480)
	bDef  = pm.BallDef(fColor=pm.Color(0.5,0.5,0.5), radius=20)

	xLength, yLength = 550, 400
	wallHorDef = pm.WallDef(sz=gm.Point(xLength, wThick), fColor=pm.Color(0.5,0.5,0.5))
	wallVerDef = pm.WallDef(sz=gm.Point(wThick, yLength), fColor=pm.Color(0.5,0.5,0.5))

	xLeft, yTop = 30, 30
	world.add_object(wallVerDef, initPos=gm.Point(xLeft, yTop))
	world.add_object(wallVerDef, initPos=gm.Point(xLeft + xLength -wThick, yTop))
	#Horizontal Wall
	world.add_object(wallHorDef, initPos=gm.Point(xLeft, yTop))
	world.add_object(wallHorDef, initPos=gm.Point(xLeft, yTop + yLength))
	world.add_object(bDef, initPos=gm.Point(200,200))
	im = world.generate_image()	
	return im, world	


def ball_world_step(i, model):
	model.step()
	im = model.generate_image()
	return im

##
# Create a physics simulation of one ball hitting the walls. 
def ball_world_simulation(): 
	plt.ion()
	plt.figure()
	#_,world = create_single_ball_world_gray()
	_,world = create_world_diamond()
	im = world.generate_image()
	plt.imshow(im)
	model = pm.Dynamics(world)		
	model.world_.dynamic_['ball-0'].set_velocity(gm.Point(2000,100))
	ballPos = deque()
	nPlot   = 20
	cmap    = plt.cm.ScalarMappable(cmap='jet')
	cmap.set_clim((0,nPlot))
	for i in range(100):
		im  = ball_world_step(i, model)
		plt.imshow(im)
		pos = model.get_object_position('ball-0')
		ballPos.append(pos)
		for j in range(min(i, nPlot)):
			plt.plot(ballPos[j].x(), ballPos[j].y(), '.',color=cmap.to_rgba(j))
		if i >= nPlot:
			p = ballPos.popleft()
			plt.plot(p.x(), p.y(), '.',color=(1.0,1.0,1.0,1.0))
		a = raw_input()
		if a=='q':
			break

def create_multiple_ball_world_gray():
	wThick = 30
	world = pm.World(xSz=640, ySz=480)
	bDef1  = pm.BallDef(fColor=pm.Color(0.5,0.5,0.5), radius=20)
	bDef2  = pm.BallDef(fColor=pm.Color(1.0,1.0,0.0), radius=20)
	xLength, yLength = 550, 400
	wallHorDef = pm.WallDef(sz=gm.Point(xLength, wThick), fColor=pm.Color(0.5,0.5,0.5))
	wallVerDef = pm.WallDef(sz=gm.Point(wThick, yLength), fColor=pm.Color(0.5,0.5,0.5))
	xLeft, yTop = 30, 30
	world.add_object(wallVerDef, initPos=gm.Point(xLeft, yTop))
	world.add_object(wallVerDef, initPos=gm.Point(xLeft + xLength -wThick, yTop))
	#Horizontal Wall
	world.add_object(wallHorDef, initPos=gm.Point(xLeft, yTop))
	world.add_object(wallHorDef, initPos=gm.Point(xLeft, yTop + yLength))
	world.add_object(bDef1, initPos=gm.Point(200,200))
	world.add_object(bDef2, initPos=gm.Point(400,200))
	im = world.generate_image()	
	return im, world	

def multi_ball_world_simulation(): 
	plt.ion()
	plt.figure()
	_,world = create_multiple_ball_world_gray()
	im = world.generate_image()
	plt.imshow(im)
	model = pm.Dynamics(world)		
	model.world_.dynamic_['ball-0'].set_velocity(gm.Point(1000,0))
	model.world_.dynamic_['ball-1'].set_velocity(gm.Point(-2000,500))
	#model.world_.dynamic_['ball-1'].set_velocity(gm.Point(0,0))
	for i in range(15):
		im  = ball_world_step(i, model)
		name = imName % i
		scm.imsave(name, im)
		#plt.imshow(im)
		#pos = model.get_object_position('ball-0')
		#ballPos.append(pos)
		#for j in range(min(i, nPlot)):
		#	plt.plot(ballPos[j].x(), ballPos[j].y(), '.',color=cmap.to_rgba(j))
		#if i >= nPlot:
		#	p = ballPos.popleft()
		#	plt.plot(p.x(), p.y(), '.',color=(1.0,1.0,1.0,1.0))
		#a = raw_input()
		#if a=='q':
		#	break

##
# Get the data for some horizon
def get_horizon_data():
	_,world = create_single_ball_world_gray()
	model   = pm.Dynamics(world)
	model.apply_force('ball-0', gm.Point(-50000, 10000), forceT=1.0)
	hor     = pm.DynamicsHorizon(model, lookAhead=5)
	for i in range(20):
		im, outMat = hor.get_data()
		print outMat

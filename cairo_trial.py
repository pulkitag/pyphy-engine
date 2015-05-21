import cairo
import numpy as np
import primitives as pm

##
#Try to paste a ball at a certain location
def paste_ball():
	bDef  = pm.BallDef()
	ball1 = pm.Ball.from_def(bDef, 'ball1', pm.Point(70,50))

	#Create the base to paste it on
	data    = np.zeros((200, 200, 4), dtype=np.uint8)
	surface = cairo.ImageSurface.create_for_data(data, 
							cairo.FORMAT_ARGB32, 200, 200)
	cr      = cairo.Context(surface)
	cr.set_source_rgba(1.0, 1.0, 1.0, 1.0)
	cr.paint()

	#Create imSurface
	shp = ball1.data_.im.shape
	y, x  = ball1.pos_.y() - ball1.yOff_, ball1.pos_.x() - ball1.xOff_
	imDat    = np.zeros((200, 200, 4), dtype=np.uint8)
	imDat[y:y+shp[0],x:x+shp[0],:] = ball1.data_.im[:] 
	surface = cairo.ImageSurface.create_for_data(imDat, 
							cairo.FORMAT_ARGB32, 200, 200)
	cr.set_source_surface(surface)		
	#cr.set_source_rgb(1.0,0.0,0.0)		
	cr.rectangle(x, y, shp[0], shp[1])
	cr.fill()
	print y,x,shp[0],shp[1]
	return data	

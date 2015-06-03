import numpy as np
import matplotlib.pyplot as plt
import collections as co
import cairo
import math
import pdb
import copy
from collections import deque
import os
import scipy.io as sio
import scipy.misc as scm
#Custom packages
import primitives as pm
import geometry as gm
import physics as phy
import os

class DataSaver:	
	def __init__(self, rootPath='/work5/pulkitag/projPhysics', numBalls=1,
							 mnBallSz=15, mxBallSz=35,
							 mnSeqLen=10, mxSeqLen=100, 
							 mnForce=1e+3, mxForce=1e+6, wThick=30,
							 isRect=True, wTheta=30, mxWLen=600, mnWLen=200,
							 arenaSz=667):
		'''
			isRect  : If the walls need to be rectangular
			wTheta  : If the walls are NOT rectangular then at what angles should they be present.
			mxWLen  : Maximum length of the walls
			mnWLen  : Minimum length of the walls
			arenaSz : Size of the arena  
		'''
		#The name of the experiment. 
		self.expStr_   = 'aSz%d_wLen%d-%d_nb%d_bSz%d-%d_f%.0e-%.0e_sLen%d-%d_wTh%d' % (arenaSz,
										  mnWLen, mxWLen, numBalls, mnBallSz, mxBallSz, mnForce, mxForce,
										  mnSeqLen, mxSeqLen, wThick)
		if not isRect:
			if isinstance(wTheta, list):
				thetaStr = '_wTheta'.join('%d-' % th for th in wTheta)
				thetaStr = 	thetaStr[0:-1]
			else:
				thetaStr = '_wTheta%d' % wTheta
			self.expStr_ = self.expStr_ + thetaStr
		#Setup directories.
		self.dirName_  = os.path.join(rootPath, self.expStr_)
		if not os.path.exists(self.dirName_):
			os.makedirs(self.dirName_)
		self.seqDir_   = os.path.join(self.dirName_, 'seq%06d')
		self.mnSeqLen_   = mnSeqLen
		self.mxSeqLen_   = mxSeqLen
		self.imFile_   = 'im%06d.jpg'
		self.dataFile_ = 'data.mat'
		#Setup variables.  
		self.numBalls_ = numBalls
		self.bmn_      = mnBallSz
		self.bmx_      = mxBallSz
		self.fmn_      = mnForce
		self.fmx_      = mxForce
		self.wlmx_     = mxWLen
		self.wlmn_     = mnWLen
		self.xSz_      = arenaSz
		self.ySz_      = arenaSz
		self.wth_      = wThick
		self.isRect_   = isRect
		if not isinstance(wTheta, list):
			wTheta = [wTheta]
		self.wTheta_   = wTheta	

	def save(self, numSeq=10):
		for i in range(numSeq):
			print i
			seqLen = int(self.mnSeqLen_ + np.random.rand() * (self.mxSeqLen_ - self.mnSeqLen_))
			self.seqLen_ = seqLen
			seqDir = self.seqDir_ % i
			if not os.path.exists(seqDir):
				os.makedirs(seqDir)
			dataFile = os.path.join(seqDir, self.dataFile_)
			imFile   = os.path.join(seqDir, self.imFile_) 
			self.save_sequence(dataFile, imFile)

	def save_sequence(self, dataFile, imFile):
		model, fx, fy = self.generate_model()
		force    = np.zeros((2 * self.numBalls_, self.seqLen_)).astype(np.float32)
		position = np.zeros((2 * self.numBalls_, self.seqLen_)).astype(np.float32)
		if self.numBalls_ > 1:
			raise Exception ('this only works with one ball for now')
		force[0,0], force[1,0] = fx, fy 
		for i in range(self.seqLen_):
			model.step()
			im = model.generate_image()
			svImFile = imFile % i
			scm.imsave(svImFile, im)				
			for j in range(self.numBalls_):
				ballName = 'ball-%d' % j
				ball     = model.get_object(ballName)
				pos      = ball.get_position()
				position[2*j,   i] = pos.x()
				position[2*j+1, i] = pos.y()
		sio.savemat(dataFile, {'force': force, 'position': position})	

	def generate_model(self):
		#Get the coordinates of the top point
		#Create the world
		self.world_ = pm.World(xSz=self.xSz_, ySz=self.ySz_)
		#Add the walls
		self.add_walls()
		#Add the balls
		self.add_balls()
		#Create physics simulation
		model = pm.Dynamics(self.world_)
		#Apply initial forces and return the result. 	
		return self.apply_force(model)	

	#This is mostly due to legacy reasons. 
	def add_rectangular_walls(self, fColor=pm.Color(1.0, 0.0, 0.0)):
		#Define the extents within which walls can be put. 
		hLen   = np.floor(self.wlmn_ + np.random.rand() * (self.wlmx_ - self.wlmn_))
		vLen   = np.floor(self.wlmn_ + np.random.rand() * (self.wlmx_ - self.wlmn_))
		topXmx = self.xSz_ - (hLen + self.wth_)
		topYmx = self.ySz_ - (vLen + self.wth_)
		xLeft = np.floor(np.random.rand() * topXmx)
		yTop  = np.floor(np.random.rand() * topYmx)	
		#Define the walls
		wallHorDef = pm.WallDef(sz=gm.Point(hLen, self.wth_), fColor=fColor)
		wallVerDef = pm.WallDef(sz=gm.Point(self.wth_, vLen), fColor=fColor)
		#Add the walls
		self.world_.add_object(wallVerDef, initPos=gm.Point(xLeft, yTop))
		self.world_.add_object(wallVerDef, initPos=gm.Point(xLeft + hLen - self.wth_, yTop))
		self.world_.add_object(wallHorDef, initPos=gm.Point(xLeft, yTop))
		self.world_.add_object(wallHorDef, initPos=gm.Point(xLeft, yTop + vLen))
		self.pts = [gm.Point(xLeft, yTop)]
		self.whl_, self.wvl_ = hLen, vLen

	def sample_walls(self):
		#For adding diagonal walls
		#1. Estimate the x and y extents of the wall. 
		#2. Find the appropriate starting position based on that
		#Sample the theta
		perm = np.random.permutation(len(self.wTheta_))
		wTheta = self.wTheta_[perm[0]]
		rad  = (wTheta * np.pi)/180.0
		hLen   = self.wlmn_ + np.random.rand() * (self.wlmx_ - self.wlmn_)
		if wTheta == 90:
			xLen = hLen
			yLen = hLen
		else:
			xLen = hLen * np.cos(rad)
			yLen = hLen * np.sin(rad)
		xExtent = 2 * xLen +  2 * self.wth_
		yExtent = 2 * yLen +  2 * self.wth_
		xLeftMin = self.wth_  
		xLeftMax = self.xSz_ - xExtent
		yLeftMin  = yLen + self.wth_
		if wTheta == 90:
			yLeftMax = self.ySz_ - self.wth_
		else:
			yLeftMax  = self.ySz_ - (yLen + self.wth_)
		#Keep sampling until the appropriate size has been found. 
		if xLeftMin <= 0 or yLeftMin <=0:
			return self.sample_walls()
		if xLeftMax < xLeftMin or yLeftMax < yLeftMin:
			return self.sample_walls()
		xLeft    = xLeftMin + np.floor(np.random.rand() * (xLeftMax - xLeftMin))
		yLeft    = yLeftMin  + np.floor(np.random.rand() * (yLeftMax - yLeftMin))	
		return xLeft, yLeft, wTheta, hLen	

	def add_walls(self):
		if self.isRect_:
			self.add_rectangular_walls()
			return
		xLeft, yLeft, wTheta, hLen = self.sample_walls()
		#Get the coordinates for creating the boundaries.  
		pt1      = gm.Point(xLeft, yLeft)
		dir1     = gm.theta2dir(-wTheta)
		pt2      = pt1 + (hLen * dir1)
		dir2     = gm.theta2dir(wTheta)
		pt3      = pt2 + (hLen * dir2)
		theta3   = (180 - wTheta)
		dir3     = gm.theta2dir(theta3)
		pt4      = pt3 + (hLen * dir3)
		pts      = [pt1, pt2, pt3, pt4]
		print "Points: ", pt1, pt2, pt3, pt4
		walls    = pm.create_cage(pts, wThick = self.wth_)	
		for w in walls:
			self.world_.add_object(w)

		#Get the lines within which the balls need to be added. 
		self.pts    = pts
		self.lines_ = []
		for i in range(len(pts)):
			self.lines_.append(gm.Line(pts[i], pts[np.mod(i+1, len(pts))]))		


	def find_point_within_lines(self, minDist):
		'''
			Find a point within the lines which is atleast minDist
			from all the boundaries. 
		'''		
		x = int(np.round(self.pts[0].x() + np.random.rand()*(self.pts[2].x() - self.pts[0].x())))
		y = int(np.round(self.pts[1].y() + np.random.rand()*(self.pts[3].y() - self.pts[1].y())))
		pt = gm.Point(x,y)
		isInside = True
		dist = []
		for (i,l) in enumerate(self.lines_):
			#Note we are finding the signed distance
			dist.append(l.distance_to_point(pt))
			if dist[i] <= minDist:
				isInside=False
		md = min(dist)
		return pt, isInside, md
					
	#Generates and adds the required number of balls. 
	def add_balls(self):
		#Generate ball definitions
		bDefs = []
		for i in range(self.numBalls_):
			#Randomly sample the radius of the ball
			r    = int(np.floor(self.bmn_ + np.random.rand() * (self.bmx_ - self.bmn_))) 
			bDef = pm.BallDef(radius=r, fColor=pm.Color(0.5, 0.5, 0.5))
			#Find a position to keep the ball
			if self.isRect_:
				xLeft, yTop = self.pts[0].x_asint(), self.pts[0].y_asint()
				xMn  = xLeft + 2 * r + self.wth_
				yMn  = yTop  + 2 * r + self.wth_
				xMx  = xLeft + self.whl_ - self.wth_ - 2 * r
				yMx  = yTop  + self.wvl_ - self.wth_ - 2 * r
				xLoc = int(np.floor(xMn + (xMx - xMn) * np.random.rand()))
				yLoc = int(np.floor(yMn + (yMx - yMn) * np.random.rand()))
			else:
				findFlag = True
				count    = 0
				while findFlag:
					pt, isValid, md = self.find_point_within_lines(r + self.wth_ + 2) #2 is safety margin	
					count += 1
					if isValid:
						findFlag=False
					if count >= 500:
						print "Failed to find a point to place the ball"
						pdb.set_trace()
				print "Ball at (%f, %f), dist: %f" % (pt.x(), pt.y(), md)
				xLoc, yLoc = pt.x_asint(), pt.y_asint()	
			self.world_.add_object(bDef, initPos=gm.Point(xLoc, yLoc))

	#Apply intial forces on the balls
	def apply_force(self, model):
		fx, fy = None, None
		for i in range(self.numBalls_):
			ballName = 'ball-%d' % i
			fx = self.fmn_ + np.floor(np.random.rand()*(self.fmx_ - self.fmn_))			
			fy = self.fmn_ + np.floor(np.random.rand()*(self.fmx_ - self.fmn_))			
			if np.random.rand() > 0.5:
				fx = -fx
			if np.random.rand() > 0.5:
				fy = -fy
			f  = gm.Point(fx, fy)
			model.apply_force(ballName, f, forceT=1.0) 
		return model, fx, fy


def save_nonrect_arena_val(numSeq=100):
	sv = DataSaver(wThick=20, isRect=False, mxForce=1e+5, wLen=300,
								 mnSeqLen=10, mxSeqLen=100, wTheta=[23, 38, 45, 53])
	sv.save(numSeq=numSeq)	

def save_nonrect_arena_train(numSeq=10000):
	drName = '/data1/pulkitag/projPhysics/'
	sv = DataSaver(rootPath=drName, wThick=20, isRect=False, mnForce=5e+4, mxForce=5e+5, 
								 mnWLen=200, mxWLen=500,
								 mnSeqLen=10, mxSeqLen=10, mnBallSz=25, mxBallSz=25, wTheta=[30, 60], arenaSz=1000)
	sv.save(numSeq=numSeq)	


def save_rect_arena(numSeq=100):
	sv = DataSaver(wThick=20, isRect=True, mxForce=1e+5, wLen=300,
								 mnSeqLen=10, mxSeqLen=100)
	sv.save(numSeq=numSeq)	

def save_multishape_rect_arena(numSeq=1000):
	sv = DataSaver(wThick=20, isRect=True, mnForce=5e+4, mxForce=5e+5, mnWLen=120, mxWLen=200,
								 mnSeqLen=10, mxSeqLen=100, mnBallSz=25, mxBallSz=25)
	sv.save(numSeq=numSeq)	


def delete_garbage():
	datDir = '/work5/pulkitag/projPhysics/aSz667_wLen300_nb1_bSz15-35_f1e+03-1e+05_sLen10-100_wTh2030-_wTheta60/seq%06d/'
	for i in range(6000):
		seqDir  = datDir % (i)
		imFile  = seqDir + 'im%06d.jpg'
		datFile = os.path.join(seqDir, 'data.mat')
		dat     = sio.loadmat(datFile, squeeze_me=True)
		N       = dat['position'].shape[1] 		
		for j in range(N,100):
			imName = imFile % j
			if os.path.exists(imName):
				os.remove(imName)		

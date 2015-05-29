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

class DataSaver:	
	def __init__(self, rootPath='/work5/pulkitag/projPhysics', numBalls=1,
							 mnBallSz=15, mxBallSz=35, seqLen=100, 
							 mnForce=1e+3, mxForce=1e+6, wThick=30,
							 isRect=False, wTheta=30, wLen=600,
							 arenaSz=667):
		'''
			isRect  : If the walls need to be rectangular
			wTheta  : If the walls are NOT rectangular then at what angles should they be present.
			wLen    : Lenght of the walls
			arenaSz : Size of the arena  
		'''
		#The name of the experiment. 
		self.expStr_   = 'aSz%d_wLen%d_nb%d_bSz%d-%d_f%.0e-%.0e_sLen%d_wTh%d' % (arenaSz,
										  numBalls, wLen, mnBallSz, mxBallSz, mnForce, mxForce, seqLen,
											wThick)
		if isRect:
			self.expStr_ = self.expStr_ + '_wTheta%d' % wTheta
		#Setup directories.
		self.dirName_  = os.path.join(rootPath, self.expStr_)
		if not os.path.exists(self.dirName_):
			os.makedirs(self.dirName_)
		self.seqDir_   = os.path.join(self.dirName_, 'seq%06d')
		self.seqLen_   = seqLen
		self.imFile_   = 'im%06d.jpg'
		self.dataFile_ = 'data.mat'
		#Setup variables.  
		self.numBalls_ = numBalls
		self.bmn_      = mnBallSz
		self.bmx_      = mxBallSz
		self.fmn_      = mnForce
		self.fmx_      = mxForce
		self.whl_      = wLen #Horizontal wall length
		self.wvl_      = wLen #Vertical wall length
		self.xSz_      = arenaSz
		self.ySz_      = arenaSz
		self.wth_      = wThick
		self.isRect_   = isRect
		self.wTheta_   = wTheta	

	def save(self, numSeq=10):
		for i in range(numSeq):
			print i
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
		return self.apply_force(model)	

	#This is mostly due to legacy reasons. 
	def add_rectangular_walls(self):
		#Define the extents within which walls can be put. 
		topXmx = self.xSz_ - (self.whl_ + self.wth_)
		topYmx = self.ySz_ - (self.wvl_ + self.wth_)
		xLeft = np.floor(np.random.rand() * topXmx)
		yTop  = np.floor(np.random.rand() * topYmx)	
		#Define the walls
		wallHorDef = pm.WallDef(sz=gm.Point(self.whl_, self.wth_), fColor=pm.Color(0.5,0.5,0.5))
		wallVerDef = pm.WallDef(sz=gm.Point(self.wth_, self.wvl_), fColor=pm.Color(0.5,0.5,0.5))
		#Add the walls
		self.world_.add_object(wallVerDef, initPos=gm.Point(xLeft, yTop))
		self.world_.add_object(wallVerDef, initPos=gm.Point(xLeft + self.whl_ - self.wth_, yTop))
		self.world_.add_object(wallHorDef, initPos=gm.Point(xLeft, yTop))
		self.world_.add_object(wallHorDef, initPos=gm.Point(xLeft, yTop + self.wvl_))

	def add_walls(self):
		if self.isRect_:
			self.add_rectangular_walls()
			return
		#For adding diagonal walls
		#1. Estimate the x and y extents of the wall. 
		#2. Find the appropriate starting position based on that
		rad  = (self.wTheta_ * np.pi)/360.0
		xLen = self.whl_ * np.cos(rad)
		yLen = self.whl_ * np.sin(rad)
		xExtent = 2 * xLen +  2 * wThick
		yExtent = 2 * yLen + 2 * wThick
		xLeftMin = wThick  
		xLeftMax = self.xSz_ - xExtent
		yLeftMin  = yLen + wThick
		yLeftMax  = self.ySz_ - yExtent 
		assert xLeftMax >= xLeftMin and yLeftMax >= yLeftMin, "Size ranges are inappropriate"
		xLeft    = xLeftMin + np.floor(np.random.rand() * (xLeftMax - xLeftMin))
		yLeft    = yLeftMin  + np.floor(np.random.rand() * (yLeftMax - yLeftMin))	

	
	#Generates and adds the required number of balls. 
	def add_balls(self):
		#Generate ball definitions
		bDefs = []
		for i in range(self.numBalls_):
			r    = int(np.floor(self.bmn_ + np.random.rand() * (self.bmx_ - self.bmn_))) 
			bDef = pm.BallDef(radius=r, fColor=pm.Color(0.5, 0.5, 0.5))
			#Find a position to keep the ball
			xMn  = xLeft + 2 * r + self.wth_
			yMn  = yTop  + 2 * r + self.wth_
			xMx  = xLeft + self.whl_ - self.wth_ - 2 * r
			yMx  = yTop  + self.wvl_ - self.wth_ - 2 * r
			xLoc = int(np.floor(xMn + (xMx - xMn) * np.random.rand()))
			yLoc = int(np.floor(yMn + (yMx - yMn) * np.random.rand()))
			self.world_.add_object(bDef, initPos=gm.Point(xLoc, yLoc))

	#Apply intial forces on the balls
	def apply_force(self, model):
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


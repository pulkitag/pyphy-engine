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
							 mnForce=1e+3, mxForce=1e+6, wThick=30):
		self.expStr_   = 'nb%d_bSz%d-%d_f%.0e-%.0e_sLen%d_wTh%d' % (numBalls, 
											mnBallSz, mxBallSz, mnForce, mxForce, seqLen,
											wThick)
		self.dirName_  = os.path.join(rootPath, self.expStr_)
		if not os.path.exists(self.dirName_):
			os.makedirs(self.dirName_)
		self.seqDir_   = os.path.join(self.dirName_, 'seq%06d')
		self.seqLen_   = seqLen
		self.imFile_   = 'im%06d.jpg'
		self.dataFile_ = 'data.mat' 
		self.numBalls_ = numBalls
		self.bmn_      = mnBallSz
		self.bmx_      = mxBallSz
		self.fmn_      = mnForce
		self.fmx_      = mxForce
		self.whl_      = 600 #Horizontal wall length
		self.wvl_      = 600 #Vertical wall length
		self.xSz_      = 667
		self.ySz_      = 667
		self.wth_      = wThick
		
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
		topXmx = self.xSz_ - (self.whl_ + self.wth_)
		topYmx = self.ySz_ - (self.wvl_ + self.wth_)
		xLeft = np.floor(np.random.rand() * topXmx)
		yTop  = np.floor(np.random.rand() * topYmx)	
		#Create the world
		world = pm.World(xSz=self.xSz_, ySz=self.ySz_)
		#Define the walls
		wallHorDef = pm.WallDef(sz=gm.Point(self.whl_, self.wth_), fColor=Color(0.5,0.5,0.5))
		wallVerDef = pm.WallDef(sz=gm.Point(self.wth_, self.wvl_), fColor=Color(0.5,0.5,0.5))
		#Add the walls
		world.add_object(wallVerDef, initPos=gm.Point(xLeft, yTop))
		world.add_object(wallVerDef, initPos=gm.Point(xLeft + self.whl_ - self.wth_, yTop))
		world.add_object(wallHorDef, initPos=gm.Point(xLeft, yTop))
		world.add_object(wallHorDef, initPos=gm.Point(xLeft, yTop + self.wvl_))
		#Generate ball definitions
		bDefs = []
		for i in range(self.numBalls_):
			r    = int(np.floor(self.bmn_ + np.random.rand() * (self.bmx_ - self.bmn_))) 
			bDef = pm.BallDef(radius=r, fColor=Color(0.5, 0.5, 0.5))
			#Find a position to keep the ball
			xMn  = xLeft + 2 * r + self.wth_
			yMn  = yTop  + 2 * r + self.wth_
			xMx  = xLeft + self.whl_ - self.wth_ - 2 * r
			yMx  = yTop  + self.wvl_ - self.wth_ - 2 * r
			xLoc = int(np.floor(xMn + (xMx - xMn) * np.random.rand()))
			yLoc = int(np.floor(yMn + (yMx - yMn) * np.random.rand()))
			world.add_object(bDef, initPos=gm.Point(xLoc, yLoc))
		#Create physics simulation
		model = pm.Dynamics(world)	
		#Apply initial forces to balls
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



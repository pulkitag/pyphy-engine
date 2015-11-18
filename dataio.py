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
import pickle
#Custom packages
import primitives as pm
import geometry as gm
import physics as phy
import os
from os import path as osp

class DataSaver:	
	def __init__(self, rootPath='/work5/pulkitag/projPhysics', numBalls=1,
							 mnBallSz=15, mxBallSz=35,
							 mnSeqLen=40, mxSeqLen=100, 
							 mnForce=1e+3, mxForce=1e+6, wThick=30,
							 isRect=True, wTheta=30, mxWLen=600, mnWLen=200,
							 arenaSz=667, oppForce=False,
							 svPrefix=None, randSeed=None, verbose=0, **kwargs):
		'''
			isRect  : If the walls need to be rectangular
			wTheta  : If the walls are NOT rectangular then at what angles should they be present.
			mxWLen  : Maximum length of the walls
			mnWLen  : Minimum length of the walls
			arenaSz : Size of the arena 
			svPrefix: Prefix in the file names for saving the data 
		'''
		#print (rootPath)
		#The name of the experiment. 
		self.expStr_   = 'aSz%d_wLen%d-%d_nb%d_bSz%d-%d_f%.0e-%.0e_sLen%d-%d_wTh%d' % (arenaSz,
										  mnWLen, mxWLen, numBalls, mnBallSz, mxBallSz, mnForce, mxForce,
										  mnSeqLen, mxSeqLen, wThick)
		if svPrefix is not None:
			self.expStr_ = svPrefix + '-' + self.expStr_

		if not isRect:
			if isinstance(wTheta, list):
				thetaStr = '_wTheta'.join('%d-' % th for th in wTheta)
				thetaStr = 	thetaStr[0:-1]
			else:
				thetaStr = '_wTheta%d' % wTheta
			self.expStr_ = self.expStr_ + thetaStr
		if oppForce:
			self.expStr_   = self.expStr_ + '_oppFrc'	

		#pdb.set_trace()
		#Setup directories.
		self.dirName_  = os.path.join(rootPath, self.expStr_)
		if not os.path.exists(self.dirName_):
			os.makedirs(self.dirName_)
		self.seqDir_   = os.path.join(self.dirName_, 'seq%06d')
		self.mnSeqLen_   = mnSeqLen
		self.mxSeqLen_   = mxSeqLen
		self.imFile_    = 'im%06d.jpg'
		self.dataFile_  = 'data.mat'
		self.worldFile_ = 'world.pkl' #Saves the world. 
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
		self.oppForce_ = oppForce
		self.verbose_  = verbose
		if not isinstance(wTheta, list):
			wTheta = [wTheta]
		self.wTheta_   = wTheta	
		if randSeed is None:
			self.rand_ = np.random.RandomState()
		else:
			self.rand_ = np.random.RandomState(randSeed)


	def save(self, numSeq=10):
		for i in range(numSeq):
			print i
			seqLen = int(self.mnSeqLen_ + self.rand_.rand() * (self.mxSeqLen_ - self.mnSeqLen_))
			self.seqLen_ = seqLen
			seqDir = self.seqDir_ % i
			if not os.path.exists(seqDir):
				os.makedirs(seqDir)
			dataFile  = os.path.join(seqDir, self.dataFile_)
			imFile    = os.path.join(seqDir, self.imFile_) 
			worldFile = os.path.join(seqDir, self.worldFile_)
			self.save_sequence(dataFile, imFile, worldFile)

	def save_sequence(self, dataFile, imFile, worldFile):
		model, f, ballPos, walls = self.generate_model()
		force    = np.zeros((2 * self.numBalls_, self.seqLen_)).astype(np.float32)
		position = np.zeros((2 * self.numBalls_, self.seqLen_)).astype(np.float32)

		#Collect all the objects in the worlds
		#objs = {}
		#for name in self.world_.get_object_names():
		#	objs[name] = self.world_.get_object(name)
		#pdb.set_trace()
		pickle.dump({'force': f, 'ballPos': ballPos, 'walls': self.pts}, open(worldFile,'w'))

		for b in range(self.numBalls_):
			fb = f[b]
			st, en = 2*b, 2*b + 1
			force[st,0], force[en,0] = fb.x(), fb.y()
			print fb 
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

	def fetch(self, cropSz=None):
		seqLen = int(self.mnSeqLen_ + self.rand_.rand() * (self.mxSeqLen_ - self.mnSeqLen_))
		self.seqLen_ = seqLen
		model, f, ballPos, walls = self.generate_model()
		force    = np.zeros((2 * self.numBalls_, self.seqLen_ - 1)).astype(np.float32)
		position = np.zeros((2 * self.numBalls_, self.seqLen_ - 1)).astype(np.float32)
		velocity = np.zeros((2 * self.numBalls_, self.seqLen_ - 1)).astype(np.float32)
		imList  = []
		imBalls = []
		for b in range(self.numBalls_):
			imBalls.append([])
			fb = f[b]
			st, en = 2*b, 2*b + 1
			force[st,0], force[en,0] = fb.x(), fb.y()
		for i in range(self.seqLen_ - 1):
			model.step()
			im = model.generate_image()
			for j in range(self.numBalls_):
				ballName = 'ball-%d' % j
				ball     = model.get_object(ballName)
				pos      = ball.get_position()
				vel      = ball.get_velocity()
				position[2*j,   i] = pos.x()
				position[2*j+1, i] = pos.y()
				velocity[2*j,   i] = vel.x()
				velocity[2*j+1, i] = vel.y()
				xMid, yMid = pos.x(), pos.y()
				if cropSz is not None:
					imBall = 255 * np.ones((cropSz, cropSz,3)).astype(np.uint8)
					x1, x2 = max(0, int(xMid - cropSz/2.0)), min(self.xSz_, int(xMid + cropSz/2.0))
					y1, y2 = max(0, int(yMid - cropSz/2.0)), min(self.ySz_, int(yMid + cropSz/2.0))
					xSz, ySz = x2 - x1, y2 - y1
					imX1 = int(cropSz/2.0) - int(np.floor(xSz/2.0))
					imX2 = int(cropSz/2.0) + int(np.ceil(xSz/2.0))
					imY1 = int(cropSz/2.0) - int(np.floor(ySz/2.0))
					imY2 = int(cropSz/2.0) + int(np.ceil(ySz/2.0))
					imBall[imY1:imY2,imX1:imX2,:] = im[y1:y2, x1:x2,0:3]
					imBalls[j].append(imBall)
			imList.append(im)
		if cropSz is None:
			return imList
		else:
			return imBalls, force, velocity 
		

	def generate_model(self):
		#Get the coordinates of the top point
		#Create the world
		self.world_ = pm.World(xSz=self.xSz_, ySz=self.ySz_)
		#Add the walls
		walls = self.add_walls()
		#Add the balls
		ballPos = self.add_balls()
		#Create physics simulation
		model = pm.Dynamics(self.world_)
		#Apply initial forces and return the result. 	
		model, fs =  self.apply_force(model)	
		return model, fs, ballPos, walls

	

	#This is mostly due to legacy reasons. 
	def add_rectangular_walls(self, fColor=pm.Color(1.0, 0.0, 0.0)):
		#Define the extents within which walls can be put. 
		hLen   = np.floor(self.wlmn_ + self.rand_.rand() * (self.wlmx_ - self.wlmn_))
		vLen   = np.floor(self.wlmn_ + self.rand_.rand() * (self.wlmx_ - self.wlmn_))
		topXmx = self.xSz_ - (hLen + self.wth_)
		topYmx = self.ySz_ - (vLen + self.wth_)
		xLeft = np.floor(self.rand_.rand() * topXmx)
		yTop  = np.floor(self.rand_.rand() * topYmx)	
		walls  = self._create_walls(xLeft, yTop, (0, 90, 180), (hLen - self.wth_, vLen, hLen - self.wth_), 
															 fColor=fColor)
	
		#Define the walls
		#wallHorDef = pm.WallDef(sz=gm.Point(hLen, self.wth_), fColor=fColor)
		#wallVerDef = pm.WallDef(sz=gm.Point(self.wth_, vLen), fColor=fColor)
		#self.world_.add_object(wallVerDef, initPos=gm.Point(xLeft, yTop))
		#self.world_.add_object(wallVerDef, initPos=gm.Point(xLeft + hLen - self.wth_, yTop))
		#self.world_.add_object(wallHorDef, initPos=gm.Point(xLeft, yTop))
		#self.world_.add_object(wallHorDef, initPos=gm.Point(xLeft, yTop + vLen))
		#self.pts = [gm.Point(xLeft, yTop)]
		#self.whl_, self.wvl_ = hLen, vLen
		return walls

	##
	def sample_walls(self):
		#For adding diagonal walls
		#1. Estimate the x and y extents of the wall. 
		#2. Find the appropriate starting position based on that
		#Sample the theta
		perm = self.rand_.permutation(len(self.wTheta_))
		wTheta = self.wTheta_[perm[0]]
		rad  = (wTheta * np.pi)/180.0
		hLen   = self.wlmn_ + self.rand_.rand() * (self.wlmx_ - self.wlmn_)
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
		xLeft    = xLeftMin + np.floor(self.rand_.rand() * (xLeftMax - xLeftMin))
		yLeft    = yLeftMin  + np.floor(self.rand_.rand() * (yLeftMax - yLeftMin))	
		return xLeft, yLeft, wTheta, hLen	

	##
	def _create_walls(self, xLeft, yLeft, thetas, wLens, fColor):
		theta1, theta2, theta3 = thetas
		wLen1,  wLen2,  wLen3  = wLens
		pt1      = gm.Point(xLeft, yLeft)
		dir1     = gm.theta2dir(theta1)
		pt2      = pt1 + (wLen1 * dir1)
		dir2     = gm.theta2dir(theta2)
		pt3      = pt2 + (wLen2 * dir2)
		dir3     = gm.theta2dir(theta3)
		pt4      = pt3 + (wLen3 * dir3)
		pts      = [pt1, pt2, pt3, pt4]
		if self.verbose_ > 0:
			print ("Points: ", pt1, pt2, pt3, pt4)
		walls    = pm.create_cage(pts, wThick = self.wth_, fColor=fColor)	
		#Get the lines within which the balls need to be added. 
		self.pts    = pts
		self.lines_ = []
		for w in walls:
			self.world_.add_object(w)
		for i in range(len(pts)):
			self.lines_.append(gm.Line(pts[i], pts[np.mod(i+1, len(pts))]))		
		return walls

	def add_walls(self, fColor=pm.Color(1.0, 0.0, 0.0)):
		if self.isRect_:
			return self.add_rectangular_walls(fColor=fColor)
		xLeft, yLeft, wTheta, hLen = self.sample_walls()
		walls = self._create_walls(xLeft, yLeft, (-wTheta, wTheta, 180-wTheta),
							 (hLen, hLen, hLen), fColor=fColor)
		return walls		


	def find_point_within_lines(self, minDist):
		'''
			Find a point within the lines which is atleast minDist
			from all the boundaries. 
		'''		
		x = int(np.round(self.pts[0].x() + self.rand_.rand()*(self.pts[2].x() - self.pts[0].x())))
		y = int(np.round(self.pts[1].y() + self.rand_.rand()*(self.pts[3].y() - self.pts[1].y())))
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
		allR, allPos = [], []
		for i in range(self.numBalls_):
			placeFlag = True
			while placeFlag:
					#Randomly sample the radius of the ball
					r    = int(np.floor(self.bmn_ + self.rand_.rand() * (self.bmx_ - self.bmn_))) 
					bDef = pm.BallDef(radius=r, fColor=pm.Color(0.5, 0.5, 0.5))
					#Find a position to keep the ball
					'''
					if self.isRect_:
						xLeft, yTop = self.pts[0].x_asint(), self.pts[0].y_asint()
						#xMn  = xLeft + 2 * r + self.wth_
						#yMn  = yTop  + 2 * r + self.wth_
						#xMx  = xLeft + self.whl_ - self.wth_ - 2 * r
						#yMx  = yTop  + self.wvl_ - self.wth_ - 2 * r
						xMn  = xLeft + r + self.wth_ + 2 #Give some margin
						yMn  = yTop  + r + self.wth_ + 2
						xMx  = xLeft + self.whl_ - self.wth_ - r - 2
						yMx  = yTop  + self.wvl_ - self.wth_ - r - 2
						xLoc = int(np.floor(xMn + (xMx - xMn) * self.rand_.rand()))
						yLoc = int(np.floor(yMn + (yMx - yMn) * self.rand_.rand()))
					else:
					'''
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
					if self.verbose_ > 0:
						print ("Ball at (%f, %f), dist: %f" % (pt.x(), pt.y(), md))
					xLoc, yLoc = pt.x_asint(), pt.y_asint()	
					pt = gm.Point(xLoc, yLoc)
					#Determine if the ball can be placed at the chosen position or not
					isOk = True
					for j in range(i):
						dist = pt.distance(allPos[j])
						if self.verbose_ > 0:
							print ("Placement Dist:", dist)
						isOk = isOk and dist > (allR[j] + r)
					if isOk:
						placeFlag = False
						allR.append(r)
						allPos.append(pt)
			self.world_.add_object(bDef, initPos=gm.Point(xLoc, yLoc))
		return allPos

	#Apply intial forces on the balls
	def apply_force(self, model):
		fs = []
		for i in range(self.numBalls_):
			ballName = 'ball-%d' % i
			if self.oppForce_:
				pos1      = self.world_.get_object_position(ballName)
				ballName2 = 'ball-%d' % np.mod(i+1,2)
				pos2      = self.world_.get_object_position(ballName2)
				ff        = pos2 - pos1 
				mag       = self.fmn_ + np.floor(self.rand_.rand()*(self.fmx_ - self.fmn_))			
				ff.make_unit_norm()
				ff.scale(mag)
				fx, fy = ff.x(), ff.y()
				if self.verbose_ > 0:
					print (fx, fy)
				print ('LOC 1')
			else:
				rnd1, rnd2 = self.rand_.rand(), self.rand_.rand()
				fDiff      = self.fmx_ - self.fmn_
				print ('LOC 2 - %f, %f' % (rnd1 * fDiff, rnd2 * fDiff))
				print ('Min/Max - %f, %f' % (self.fmx_, self.fmn_))
				#Sample magnitude
				fMag   = self.fmn_ + np.floor(rnd1 * (self.fmx_ - self.fmn_))
				#Sample Theta
				fTheta = rnd1 * np.pi 			
				if self.rand_.rand() > 0.5:
					fTheta = -fTheta
				fx, fy = fMag * np.cos(fTheta), fMag * np.sin(fTheta)
			print ('FORCE - FX: %f, FY: %f' % (fx, fy))
			f  = gm.Point(fx, fy)
			model.apply_force(ballName, f, forceT=1.0) 
			fs.append(f)
		return model, fs


def save_nonrect_arena_val(numSeq=100):
	sv = DataSaver(wThick=20, isRect=False, mxForce=1e+5, wLen=300,
								 mnSeqLen=10, mxSeqLen=100, wTheta=[23, 38, 45, 53])
	sv.save(numSeq=numSeq)	

def save_nonrect_arena_train(numSeq=10000, oppForce=False, numBalls=1, svPrefix=None, 
														 mnWLen=500, mxWLen=800, arenaSz=1600, mnForce=3e+4, 
														 mxForce=8e+4):
	drName = '/work5/pulkitag/projPhysics/'
	sv = DataSaver(rootPath=drName, wThick=30, isRect=False, mnForce=mnForce, mxForce=mxForce, 
								 mnWLen=mnWLen, mxWLen=mxWLen, numBalls=numBalls,
								 mnSeqLen=10, mxSeqLen=200, mnBallSz=25, mxBallSz=25, wTheta=[30, 60],
								 arenaSz=arenaSz, svPrefix=svPrefix)
	sv.save(numSeq=numSeq)	


def save_rect_arena(numSeq=10000, oppForce=False, numBalls=1, svPrefix=None,
										mnForce=3e+4, mxForce=8e+4, mnWLen=300, mxWLen=550, arenaSz=700,
										mnSeqLen=10, mxSeqLen=200):

	drName = '/data0/pulkitag/projPhysics/'
	sv = DataSaver(wThick=30, isRect=True, mnForce=mnForce, mxForce=mxForce, 
								 mnWLen=mnWLen, mxWLen=mxWLen, mnSeqLen=mnSeqLen, mxSeqLen=mxSeqLen,
								 numBalls=numBalls, mnBallSz=25, mxBallSz=25, arenaSz=arenaSz,
								 oppForce=oppForce, svPrefix=svPrefix, rootPath=drName)
	sv.save(numSeq=numSeq)	

def save_multishape_rect_arena(numSeq=1000, numBalls=1, oppForce=False):
	drName = '/data1/pulkitag/projPhysics/'
	sv = DataSaver(rootPath=drName,numBalls=numBalls,wThick=20, isRect=True, mnForce=1e+4,
								 mxForce=1e+5, mnWLen=400, mxWLen=400,
								 mnSeqLen=40, mxSeqLen=40, mnBallSz=25, mxBallSz=25, oppForce=oppForce)
	sv.save(numSeq=numSeq)	


def stats_force():
	#dataDir = '/work5/pulkitag/projPhysics/trainV2-aSz700_wLen300-550_nb1_bSz25-25_f3e+04-8e+04_sLen10-200_wTh30/'
	dataDir = '/work5/pulkitag/projPhysics/trainV2-aSz700_wLen200-550_nb2_bSz25-25_f8e+03-5e+04_sLen10-200_wTh30_oppFrc'
	theta = []
	for i in range(10000):
		if np.mod(i,1000)==1:
			print(i)
		seqFolder = osp.join(dataDir, 'seq%06d' % i)
		wFile     = osp.join(seqFolder, 'data.mat')
		data      = sio.loadmat(wFile)
		force     = data['force'][:,0]
		theta.append(np.arctan2(float(force[0]), float(force[1])))
	return theta
		


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

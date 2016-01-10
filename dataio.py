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
	print ('DATAIO SETUP DONE')

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

	def fetch(self, cropSz=None, procSz=None):
		seqLen = int(self.mnSeqLen_ + self.rand_.rand() * (self.mxSeqLen_ - self.mnSeqLen_))
		self.seqLen_ = seqLen
		model, f, ballPos, walls = self.generate_model()
		force    = np.zeros((2 * self.numBalls_, self.seqLen_)).astype(np.float32)
		position = np.zeros((2 * self.numBalls_, self.seqLen_)).astype(np.float32)
		velocity = np.zeros((2 * self.numBalls_, self.seqLen_)).astype(np.float32)
		imList  = []
		imBalls = []
		for b in range(self.numBalls_):
			imBalls.append([])
			fb = f[b]
			st, en = 2*b, 2*b + 1
			force[st,0], force[en,0] = fb.x(), fb.y()
		#Previous position
		pPos = np.nan * np.zeros((self.numBalls_,2))
		for i in range(self.seqLen_):
			model.step()
			im = model.generate_image()
			vx, vy = None, None
			for j in range(self.numBalls_):
				ballName = 'ball-%d' % j
				ball     = model.get_object(ballName)
				pos      = ball.get_position()
				position[2*j,   i] = pos.x()
				position[2*j+1, i] = pos.y()
				#Speed should not be predicted, instead we should just predict
				#delta in position. The difference between the two is critical 
				#due to collisions. 
				if not np.isnan(pPos[j][0]):
					vx = pos.x() - pPos[j][0]
					vy = pos.y() - pPos[j][1]
				pPos[j][0], pPos[j][1] = pos.x(), pos.y()
				xMid, yMid = round(pos.x()), round(pos.y())
				if cropSz is not None:
					imBall = 255 * np.ones((cropSz, cropSz,3)).astype(np.uint8)
					#Cropping coordinates in the original image
					x1, x2 = max(0, xMid - cropSz/2.0), min(self.xSz_, xMid + cropSz/2.0)
					y1, y2 = max(0, yMid - cropSz/2.0), min(self.ySz_, yMid + cropSz/2.0)
					#Coordinates in the cropped image centerd at the ball
					imX1 = int(round(cropSz/2.0  - (xMid - x1)))
					imX2 = int(round(cropSz/2.0  + (x2 - xMid)))
					imY1 = int(round(cropSz/2.0  - (yMid - y1)))
					imY2 = int(round(cropSz/2.0  + (y2 - yMid)))
					x1, x2 = int(round(x1)), int(round(x2))
					y1, y2 = int(round(y1)), int(round(y2))
					imBall[imY1:imY2,imX1:imX2,:] = im[y1:y2, x1:x2,0:3]
					position[2*j, i] = position[2*j, i] - x1 + imX1 
					position[2*j+1, i] = position[2*j+1, i] - y1 + imY1 

					if procSz is not None:
						posScale  = float(procSz)/float(cropSz)
						imBall = scm.imresize(imBall, (procSz, procSz))
						position[2*j, i]   = position[2*j, i] * posScale
						position[2*j+1, i] = position[2*j+1, i] * posScale
						if vx is not None:
							velocity[2*j, i-1]   = vx * posScale
							velocity[2*j+1, i-1] = vy * posScale
					imBalls[j].append(imBall)
			imList.append(im)
		if cropSz is None:
			return imList
		else:
			return imBalls, force[:, 0:self.seqLen_],\
						 velocity[:, 0:self.seqLen_],\
						 position[:, 0:self.seqLen_] 
		
	def _generate_model(self):
		#get the coordinates of the top point
		#create the world
		self.world_ = pm.World(xSz=self.xSz_, ySz=self.ySz_)
		#add the walls
		walls = self.add_walls()
		#add the balls
		ballpos = self.add_balls()
		#create physics simulation
		model = pm.Dynamics(self.world_)
		return model

	def generate_model(self):
		model = self._generate_model()
		#apply initial forces and return the result. 	
		model, fs =  self.apply_force(model)	
		return model, fs, ballpos, walls

	#this is mostly due to legacy reasons. 
	def add_rectangular_walls(self, fColor=pm.Color(1.0, 0.0, 0.0)):
		#define the extents within which walls can be put. 
		hlen   = np.floor(self.wlmn_ + self.rand_.rand() * (self.wlmx_ - self.wlmn_))
		vlen   = np.floor(self.wlmn_ + self.rand_.rand() * (self.wlmx_ - self.wlmn_))
		topxmx = self.xSz_ - (hlen + self.wth_)
		topymx = self.ySz_ - (vlen + self.wth_)
		xleft = np.floor(self.rand_.rand() * topxmx)
		ytop  = np.floor(self.rand_.rand() * topymx)	
		walls  = self._create_walls(xleft, ytop, (0, 90, 180), (hlen - self.wth_, vlen, hlen - self.wth_), 
															 fColor=fColor)
	
		#define the walls
		#wallhordef = pm.walldef(sz=gm.Point(hlen, self.wth_), fColor=fColor)
		#wallverdef = pm.walldef(sz=gm.Point(self.wth_, vlen), fColor=fColor)
		#self.world_.add_object(wallverdef, initpos=gm.Point(xleft, ytop))
		#self.world_.add_object(wallverdef, initpos=gm.Point(xleft + hlen - self.wth_, ytop))
		#self.world_.add_object(wallhordef, initpos=gm.Point(xleft, ytop))
		#self.world_.add_object(wallhordef, initpos=gm.Point(xleft, ytop + vlen))
		#self.pts = [gm.Point(xleft, ytop)]
		#self.whl_, self.wvl_ = hlen, vlen
		return walls

	##
	def sample_walls(self):
		#for adding diagonal walls
		#1. estimate the x and y extents of the wall. 
		#2. find the appropriate starting position based on that
		#sample the theta
		perm = self.rand_.permutation(len(self.wTheta_))
		wtheta = self.wTheta_[perm[0]]
		rad  = (wtheta * np.pi)/180.0
		hlen   = self.wlmn_ + self.rand_.rand() * (self.wlmx_ - self.wlmn_)
		if wtheta == 90:
			xlen = hlen
			ylen = hlen
		else:
			xlen = hlen * np.cos(rad)
			ylen = hlen * np.sin(rad)
		xextent = 2 * xlen +  2 * self.wth_
		yextent = 2 * ylen +  2 * self.wth_
		xleftmin = self.wth_  
		xleftmax = self.xSz_ - xextent
		yleftmin  = ylen + self.wth_
		if wtheta == 90:
			yleftmax = self.ySz_ - self.wth_
		else:
			yleftmax  = self.ySz_ - (ylen + self.wth_)
		#keep sampling until the appropriate size has been found. 
		if xleftmin <= 0 or yleftmin <=0:
			return self.sample_walls()
		if xleftmax < xleftmin or yleftmax < yleftmin:
			return self.sample_walls()
		xleft    = xleftmin + np.floor(self.rand_.rand() * (xleftmax - xleftmin))
		yleft    = yleftmin  + np.floor(self.rand_.rand() * (yleftmax - yleftmin))	
		return xleft, yleft, wtheta, hlen	

	##
	def _create_walls(self, xleft, yleft, thetas, wlens, fColor):
		theta1, theta2, theta3 = thetas
		wlen1,  wlen2,  wlen3  = wlens
		pt1      = gm.Point(xleft, yleft)
		dir1     = gm.theta2dir(theta1)
		pt2      = pt1 + (wlen1 * dir1)
		dir2     = gm.theta2dir(theta2)
		pt3      = pt2 + (wlen2 * dir2)
		dir3     = gm.theta2dir(theta3)
		pt4      = pt3 + (wlen3 * dir3)
		pts      = [pt1, pt2, pt3, pt4]
		if self.verbose_ > 0:
			print ("points: ", pt1, pt2, pt3, pt4)
		walls    = pm.create_cage(pts, wThick = self.wth_, fColor=fColor)	
		#get the lines within which the balls need to be added. 
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
		xleft, yleft, wtheta, hlen = self.sample_walls()
		walls = self._create_walls(xleft, yleft, (-wtheta, wtheta, 180-wtheta),
							 (hlen, hlen, hlen), fColor=fColor)
		return walls		


	def find_point_within_lines(self, mindist):
		'''
			find a point within the lines which is atleast mindist
			from all the boundaries. 
		'''		
		x = int(np.round(self.pts[0].x() + self.rand_.rand()*(self.pts[2].x() - self.pts[0].x())))
		y = int(np.round(self.pts[1].y() + self.rand_.rand()*(self.pts[3].y() - self.pts[1].y())))
		pt = gm.Point(x,y)
		isinside = True
		dist = []
		for (i,l) in enumerate(self.lines_):
			#note we are finding the signed distance
			dist.append(l.distance_to_point(pt))
			if dist[i] <= mindist:
				isinside=False
		md = min(dist)
		return pt, isinside, md
					
	#generates and adds the required number of balls. 
	def add_balls(self):
		#generate ball definitions
		allr, allpos = [], []
		for i in range(self.numBalls_):
			placeflag = True
			while placeflag:
					#randomly sample the radius of the ball
					r    = int(np.floor(self.bmn_ + self.rand_.rand() * (self.bmx_ - self.bmn_))) 
					bdef = pm.BallDef(radius=r, fColor=pm.Color(0.5, 0.5, 0.5))
					#find a position to keep the ball
					'''
					if self.isrect_:
						xleft, ytop = self.pts[0].x_asint(), self.pts[0].y_asint()
						#xmn  = xleft + 2 * r + self.wth_
						#ymn  = ytop  + 2 * r + self.wth_
						#xmx  = xleft + self.whl_ - self.wth_ - 2 * r
						#ymx  = ytop  + self.wvl_ - self.wth_ - 2 * r
						xmn  = xleft + r + self.wth_ + 2 #give some margin
						ymn  = ytop  + r + self.wth_ + 2
						xmx  = xleft + self.whl_ - self.wth_ - r - 2
						ymx  = ytop  + self.wvl_ - self.wth_ - r - 2
						xloc = int(np.floor(xmn + (xmx - xmn) * self.rand_.rand()))
						yloc = int(np.floor(ymn + (ymx - ymn) * self.rand_.rand()))
					else:
					'''
					findflag = True
					count    = 0
					while findflag:
						pt, isvalid, md = self.find_point_within_lines(r + self.wth_ + 2) #2 is safety margin	
						count += 1
						if isvalid:
							findflag=False
						if count >= 500:
							print "failed to find a point to place the ball"
							pdb.set_trace()
					if self.verbose_ > 0:
						print ("ball at (%f, %f), dist: %f" % (pt.x(), pt.y(), md))
					xloc, yloc = pt.x_asint(), pt.y_asint()	
					pt = gm.Point(xloc, yloc)
					#determine if the ball can be placed at the chosen position or not
					isok = True
					for j in range(i):
						dist = pt.distance(allpos[j])
						if self.verbose_ > 0:
							print ("placement dist:", dist)
						isok = isok and dist > (allr[j] + r)
					if isok:
						placeflag = False
						allr.append(r)
						allpos.append(pt)
			self.world_.add_object(bdef, initPos=gm.Point(xloc, yloc))
		return allpos

	#apply intial forces on the balls
	def apply_force(self, model):
		fs = []
		for i in range(self.numBalls_):
			ballname = 'ball-%d' % i
			if self.oppForce_:
				pos1      = self.world_.get_object_position(ballname)
				ballname2 = 'ball-%d' % np.mod(i+1,2)
				pos2      = self.world_.get_object_position(ballname2)
				ff        = pos2 - pos1 
				mag       = self.fmn_ + np.floor(self.rand_.rand()*(self.fmx_ - self.fmn_))			
				ff.make_unit_norm()
				ff.scale(mag)
				fx, fy = ff.x(), ff.y()
				if self.verbose_ > 0:
					print (fx, fy)
				print ('loc 1')
			else:
				rnd1, rnd2 = self.rand_.rand(), self.rand_.rand()
				fdiff      = self.fmx_ - self.fmn_
				if self.verbose_ > 0:
					print ('loc 2 - %f, %f' % (rnd1 * fdiff, rnd2 * fdiff))
					print ('min/max - %f, %f' % (self.fmx_, self.fmn_))
				#sample magnitude
				fmag   = self.fmn_ + np.floor(rnd1 * (self.fmx_ - self.fmn_))
				#sample theta
				ftheta = rnd1 * np.pi 			
				if self.rand_.rand() > 0.5:
					ftheta = -ftheta
				fx, fy = fmag * np.cos(ftheta), fmag * np.sin(ftheta)
			if self.verbose_ > 0:
				print ('force - fx: %f, fy: %f' % (fx, fy))
			f  = gm.Point(fx, fy)
			model.apply_force(ballname, f, forceT=1.0) 
			fs.append(f)
		return model, fs


def save_nonrect_arena_val(numseq=100):
	sv = datasaver(wthick=20, isrect=False, mxforce=1e+5, wlen=300,
								 mnseqlen=10, mxseqlen=100, wtheta=[23, 38, 45, 53])
	sv.save(numseq=numseq)	

def save_nonrect_arena_train(numseq=10000, oppforce=False, numballs=1, svprefix=None, 
														 mnwlen=500, mxwlen=800, arenasz=1600, mnforce=3e+4, 
														 mxforce=8e+4):
	drname = '/data0/pulkitag/projphysics/'
	sv = datasaver(rootpath=drname, wthick=30, isrect=False, mnforce=mnforce, mxforce=mxforce, 
								 mnwlen=mnwlen, mxwlen=mxwlen, numballs=numballs,
								 mnseqlen=10, mxseqlen=200, mnballsz=25, mxballsz=25, wtheta=[30, 60],
								 arenasz=arenasz, svprefix=svprefix)
	sv.save(numseq=numseq)	


def save_rect_arena(numSeq=10000, oppForce=False, numBalls=1, svPrefix=None,
										mnForce=3e+4, mxForce=8e+4, mnWLen=300, mxWLen=550, arenaSz=700,
										mnSeqLen=10, mxSeqLen=200):

	drName = '/data0/pulkitag/projphysics/'
	sv = DataSaver(wThick=30, isRect=True, mnForce=mnForce, mxForce=mxForce, 
								 mnWLen=mnWLen, mxWLen=mxWLen, mnSeqLen=mnSeqLen, mxSeqLen=mxSeqLen,
								 numBalls=numBalls, mnBallSz=25, mxBallSz=25, arenaSz=arenaSz,
								 oppForce=oppForce, svPrefix=svPrefix, rootPath=drName)
	sv.save(numSeq=numSeq)	

def save_multishape_rect_arena(numseq=1000, numballs=1, oppforce=False):
	drname = '/data1/pulkitag/projphysics/'
	sv = datasaver(rootpath=drname,numballs=numballs,wthick=20, isrect=True, mnforce=1e+4,
								 mxforce=1e+5, mnwlen=400, mxwlen=400,
								 mnseqlen=40, mxseqlen=40, mnballsz=25, mxballsz=25, oppforce=oppforce)
	sv.save(numseq=numseq)	



class CustomWorld:
	def __init__(self, numballs=1, ballsz=25, wthick=30,
							 isrect=True, wtheta=30, wlen=300,
							 arenasz=667, verbose=0, ballLocs=[gm.Point(120,120)],
							 forces=[gm.Point(5e+4, 5e+4)], **kwargs):
		self.numballs_   = numballs
		self.bsz_  = ballsz
		self.bloc_ = copy.deepcopy(ballLocs)
		self.forces_  = copy.deepcopy(forces)
		self.wthick_  = wthick
		self.wtheta_  = wtheta
		self.isrect_  = isrect
		self.wlen_    = wlen
		self.asz_     = arenasz
		self.verbose_ = verbose
		self.xSz_     = arenasz
		self.ySz_     = arenasz

	def _generate_model(self):
		#get the coordinates of the top point
		#create the world
		self.world_ = pm.World(xSz=self.asz_, ySz=self.asz_)
		#add the walls
		self.walls_   = self.add_walls()
		#add the balls
		self.add_balls()
		#create physics simulation
		self.model_   = pm.Dynamics(self.world_)

	def generate_model(self):
		self._generate_model()
		self.apply_force()

	def get_glimpse(self, ballNum=None, cropSz=128):
		self.model_.step()
		im = self.model_.generate_image()
		print (im.shape)
		position = np.zeros((2*self.numballs_,))
		velocity = np.zeros((2*self.numballs_,))
		imBalls = []
		for b in range(self.numballs_):
			ballName = 'ball-%d' % b
			ball     = self.model_.get_object(ballName)
			pos      = ball.get_position()
			vel      = ball.get_velocity()
			position[2*b] = pos.x()
			position[2*b+1] = pos.y()
			velocity[2*b] = vel.x()
			velocity[2*b+1] = vel.y()
			xMid, yMid = round(pos.x()), round(pos.y())
			if cropSz is not None:
				imBall = 255 * np.ones((cropSz, cropSz,3)).astype(np.uint8)
				x1, x2 = max(0, xMid - cropSz/2.0), min(self.xSz_, xMid + cropSz/2.0)
				y1, y2 = max(0, yMid - cropSz/2.0), min(self.ySz_, yMid + cropSz/2.0)
				print xMid, x1, x2, cropSz
				#xSz, ySz = x2 - x1, y2 - y1
				imX1 = int(cropSz/2.0  - (xMid - x1))
				imX2 = int(cropSz/2.0  + (x2 - xMid))
				imY1 = int(cropSz/2.0  - (yMid - y1))
				imY2 = int(cropSz/2.0  + (y2 - yMid))
				imBall[imY1:imY2,imX1:imX2,:] = im[int(y1):int(y2), int(x1):int(x2),0:3]
				imBalls.append(imBall)
		return imBalls, position, velocity

	def apply_force(self):
		for b in range(self.numballs_):
			ballname = 'ball-%d' % b
			self.model_.apply_force(ballname, self.forces_[b], forceT=1.0) 

	def add_walls(self, xleft=20, yleft=20, hLen=None, vLen=None, 
								fColor=pm.Color(1.0, 0.0, 0.0)):
		if hLen is None:
			hLen = self.wlen_
		if vLen is None:
			vLen = self.wlen_
		wtheta = self.wtheta_
		if type(wtheta) == list:
			th1, th2, th3 = wtheta
		else:
			th1, th2, th3 = -wtheta, wtheta, 180-wtheta
		th1, th2, th3 = fix_theta_range(th1), fix_theta_range(th2), fix_theta_range(th3)
		walls = self._create_walls(xleft, yleft, (th1, th2, th3),
							 (vLen, hLen, vLen), fColor=fColor)
		return walls		

	##
	def _create_walls(self, xleft, yleft, thetas, wlens, fColor):
		theta1, theta2, theta3 = thetas
		wlen1,  wlen2,  wlen3  = wlens
		pt1      = gm.Point(xleft, yleft)
		dir1     = gm.theta2dir(theta1)
		pt2      = pt1 + (wlen1 * dir1)
		dir2     = gm.theta2dir(theta2)
		pt3      = pt2 + (wlen2 * dir2)
		dir3     = gm.theta2dir(theta3)
		pt4      = pt3 + (wlen3 * dir3)
		pts      = [pt1, pt2, pt3, pt4]
		if self.verbose_ > 0:
			print 'Wall Points: %s, %s, %s, %s' % (pt1, pt2, pt3, pt4)
		walls    = pm.create_cage(pts, wThick = self.wthick_, fColor=fColor)	
		#get the lines within which the balls need to be added. 
		self.pts    = pts
		self.lines_ = []
		for w in walls:
			self.world_.add_object(w)
		for i in range(len(pts)):
			self.lines_.append(gm.Line(pts[i], pts[np.mod(i+1, len(pts))]))		
		return walls

	#generates and adds the required number of balls. 
	def add_balls(self):
		for i in range(self.numballs_):
			bdef = pm.BallDef(radius=self.bsz_, fColor=pm.Color(0.5, 0.5, 0.5))
			self.world_.add_object(bdef, initPos=self.bloc_[i])


def fix_theta_range(theta):
	theta = np.mod(theta, 360)
	if theta > 180:
		theta = -(360 - theta) 
	return theta
	
def stats_force():
	#datadir = '/work5/pulkitag/projphysics/trainv2-asz700_wlen300-550_nb1_bsz25-25_f3e+04-8e+04_slen10-200_wth30/'
	datadir = '/work5/pulkitag/projphysics/trainv2-asz700_wlen200-550_nb2_bsz25-25_f8e+03-5e+04_slen10-200_wth30_oppfrc'
	theta = []
	for i in range(10000):
		if np.mod(i,1000)==1:
			print(i)
		seqfolder = osp.join(datadir, 'seq%06d' % i)
		wfile     = osp.join(seqfolder, 'data.mat')
		data      = sio.loadmat(wfile)
		force     = data['force'][:,0]
		theta.append(np.arctan2(float(force[0]), float(force[1])))
	return theta

		
def delete_garbage():
	datdir = '/work5/pulkitag/projphysics/asz667_wlen300_nb1_bsz15-35_f1e+03-1e+05_slen10-100_wth2030-_wtheta60/seq%06d/'
	for i in range(6000):
		seqdir  = datdir % (i)
		imfile  = seqdir + 'im%06d.jpg'
		datfile = os.path.join(seqdir, 'data.mat')
		dat     = sio.loadmat(datfile, squeeze_me=True)
		n       = dat['position'].shape[1] 		
		for j in range(n,100):
			imname = imfile % j
			if os.path.exists(imname):
				os.remove(imName)	


def test_custom_world():
	cw = CustomWorld(wtheta=[0, 90, -180], verbose=1)
	cw.generate_model()
	plt.ion()
	for i in range(100000):
		imList, pos, vel = cw.get_glimpse(cropSz=512)
		plt.imshow(imList[0])
		plt.show()
		ip = raw_input()
		if ip=='q':
			return
	return imList, pos, vel	

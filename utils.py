import pickle
import scipy.io as sio
from os import path as osp
import numpy as np
import copy
import os
import other_utils as ou
import subprocess


def detect_collision(folderName, wThick=30, ballSz=25):
	world = pickle.load(open(osp.join(folderName, 'world.pkl'),'r'))
	walls = world['walls']
	x1, y1 = walls[0].x(), walls[0].y()
	x2, y2 = walls[1].x(), walls[1].y()
	x3, y3 = walls[2].x(), walls[2].y()
	x4, y4 = walls[3].x(), walls[3].y()
	xMin, xMax = min(x1, x2, x3, x4), max(x1, x2, x3, x4)
	yMin, yMax = min(y1, y2, y3, y4), max(y1, y2, y3, y4)

	#Get the bbox st if a collision happens within it - its ball-ball,
	#otherwise its ball-wall	
	#print (xMin, xMax, yMin, yMax)
	margin = 10
	xMin = xMin + wThick + ballSz + margin
	xMax = xMax - (wThick + ballSz + margin)
	yMin = yMin + wThick + ballSz + margin
	yMax = yMax - (wThick + ballSz + margin)
	#print (xMin, xMax, yMin, yMax)

	#Get number of balls
	nb = len(world['ballPos'])

	#Get the position of the ball
	dyn = sio.loadmat(osp.join(folderName, 'data.mat'))
	pos = dyn['position']
	thresh = 0.03
	allfIdx, allbCol = [], []
	for b in range(nb):
		st = 2*b
		en = st + 2  
		posB = pos[st:en,:]
		velB = np.diff(posB)
		rawV = copy.copy(velB)
		accB = np.diff(velB)
		accB = np.sum(accB * accB, axis=0)
		velB = np.sum(velB * velB, axis=0)
		idx  = np.where(accB >= thresh * velB[0:-1])[0]
		idx  = idx + 1
		fIdx = []
		bCol = []
		for i in range(len(idx)):
			if idx[i] in fIdx:
				continue
			if i < len(idx)-1 and (idx[i] + 1 == idx[i+1]):
				fIdx.append(idx[i+1])
			else:
				fIdx.append(idx[i])
		for i in fIdx:
			p = min(i, posB.shape[1]-1)
			x, y = posB[:,p]
			#print (x,y)
			if x>xMin and x<xMax and y>yMin and y<yMax:
				bCol.append(True)
			else:
				bCol.append(False)
		allfIdx.append(fIdx)
		allbCol.append(bCol)				
	return allfIdx, allbCol


def det_collisions_all(folderName):
	fNames = [f for f in os.listdir(folderName) if osp.isdir(osp.join(folderName,f))]
	fNames = [osp.join(folderName, f) for f in fNames]
	print (len(fNames))
	colCount  = 0
	bColCount = 0
	for i,f in enumerate(fNames[0:1000]):
		if np.mod(i,1000)==1:
			print(i)
		allfIdx, allbCol = detect_collision(f)
		for b in range(len(allfIdx)):
			colCount  += len(allfIdx[b])
			bColCount += sum(np.array(allbCol[b])==True)
	return colCount, bColCount


def save_collisions_seq(folderName, isSubFolder=True):
	fNames = [f for f in os.listdir(folderName) if osp.isdir(osp.join(folderName,f))]
	srcNames  = [osp.join(folderName, f) for f in fNames]
	if isSubFolder:
		outFolder = '/' + folderName.strip('/') + '-collisions-only'
	else:
		outFolder = '/' + folderName.strip('/') + '-collisions-only-nosub'
	outNames  = [osp.join(outFolder, f) for f in fNames]
	outCount = 0
	for ifd,ofd in zip(srcNames, outNames):
		print (ifd, ofd)
		allfIdx, allbCol = detect_collision(ifd)
		inMat = sio.loadmat(osp.join(ifd, 'data.mat'))
		inIm  = osp.join(ifd, 'im%06d.jpg')
		inPos, inForce = inMat['position'], inMat['force']
		colIdx = []
		for b in range(len(allfIdx)):
			colIdx = colIdx + allfIdx[b]
		if isSubFolder:
			outCount = 0
		for c in colIdx:
			st = max(0, c - 7)
			en = min(c+7, inPos.shape[1])
			if isSubFolder:
				colFolder = osp.join(ofd, 'sub-%04d' % outCount)
			else:
				colFolder = osp.join(outFolder, 'seq%06d' % outCount)
			outIm   = osp.join(colFolder, 'im%06d.jpg')
			ou.mkdir(colFolder)
			outCount += 1 
			#Copy the images
			for i,idx in enumerate(range(st, en)):
				oIm  = outIm % i
				iIm  = inIm  % idx
				callStr = 'cp %s %s' % (iIm, oIm)
				#print (i, idx, callStr)
				subprocess.check_call([callStr], shell=True)
			#Copy the data
			oPos, oForce = inPos[:,st:en], inForce[:,st:en]
			datFile = osp.join(colFolder, 'data.mat')
			sio.savemat(datFile, {'position': oPos, 'force': oForce}) 
	return colCount, bColCount

	 

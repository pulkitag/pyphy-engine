import dataio as dio

def save(isVal=False):
	nb       = [1, 2, 3]
	oppForce = [False, False, False]
	svPrefix = 'val'
	valRandSeed = [3792, 4185, 1094] 
	for b,o,v in zip(nb, oppForce, valRandSeed):
		dio.save_rect_arena(numSeq=100, oppForce=o, numBalls=b, 
			svPrefix=svPrefix, mnForce=8e+4, mxForce=8e+4, 
			mnWLen=300, randSeed=v,
			mxWLen=550, arenaSz=700,mnSeqLen=210, mxSeqLen=210)



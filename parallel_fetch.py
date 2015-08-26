from multiprocessing import Pool
import dataio as dio
import time

def get_model(seqLen=100, numBalls=2):
	time.sleep(3)
	ds = dio.DataSaver(wThick=30, isRect=True, mnSeqLen=seqLen, mxSeqLen=seqLen,
						 numBalls=numBalls, mnBallSz=25, mxBallSz=25)
	ims = ds.fetch()
	return ims

def run_parallel(numW=4, numJobs=10):
	p = Pool(numW)
	resPool = []
	for i in range(numJobs):
		ds = dio.DataSaver(wThick=30, isRect=True, 
							mnSeqLen=100, mxSeqLen=100,
						 numBalls=2, mnBallSz=25, mxBallSz=25)
		#resPool.append(p.apply_async(get_model))
		resPool.append(p.apply_async(ds.fetch))
	print('All Processes launched')
	res = []
	for r in resPool:
		res.append(r.get())
	return res

def run_process(numJobs=10):
	prcs = []
	for i in range(numJobs):
		p = Process(target=get_model)
		p.start()
		prcs.append(p)
	print('All Processes launched')
	for p in prcs:
		res.append(r.get())
	return res

def run_serial():
	for i in range(100):
		tmpRes = get_model()
		

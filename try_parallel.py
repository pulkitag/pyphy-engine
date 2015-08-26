import numpy as np
from multiprocessing import Pool, Process
import time

def f(x):
	time.sleep(2)
	return x*x

def rand():
	return np.random.random()

def run_map(numW=4, numJobs=20):
	t1= time.time()
	pool = Pool(numW)
	print(pool.map(f, range(numJobs)))
	t2 = time.time()
	print('Time: ', t2 - t1)


def run_process(numJobs=20):
	t1= time.time()
	prcs = []
	for i in range(numJobs):
		p = Process(target=f, args=(i,))
		p.start()
		prcs.append(p)
	print('All Processes launched')
	for p in prcs:
		p.join()
	t2= time.time()
	print('Time: ', t2 - t1)

def run_async(numW=4, numJobs=20):
	t1= time.time()
	p = Pool(numW)
	resPool = []
	for i in range(numJobs):
		#resPool.append(p.apply_async(f, [i]))
		resPool.append(p.apply_async(rand))
	print('All Processes launched')
	res = []
	for r in resPool:	
		res.append(r.get())
	t2= time.time()
	print('Time: ', t2 - t1)
	p.close()
	return res
	

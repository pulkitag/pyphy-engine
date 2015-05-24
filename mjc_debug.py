import mjcpy2 as mj

def main():
	defFile = '/work4/pulkitag-code/pkgs/python-mojoco/my-ball.xml'
	world   = mj.MJCWorld2(defFile)
	img     = world.GetImage()

import numpy as np
import cv2
import glob

def getBatch(imgdir, id, batch_size, width, height):
	
	filelist = glob.glob(imgdir+'/*.jpeg')
	filelist.sort()
	n = len(filelist)

	imgs1 =[None]*batch_size
	imgs2 =[None]*batch_size

	for x in xrange(0,batch_size):
		imgs1[x] = cv2.resize( cv2.imread( filelist[ (2*id + 1 + x)%n ],0), (width, height) )[np.newaxis,:,:,np.newaxis]
		imgs2[x] = cv2.resize( cv2.imread( filelist[ (2*id + 2 + x)%n ],0), (width, height) )[np.newaxis,:,:,np.newaxis]


	return (np.concatenate(imgs1,0), np.concatenate(imgs2,0)) 




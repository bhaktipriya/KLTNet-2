import numpy as np
import cv2
import glob

def getBatch(imgdir, id, batch_size, width, height):
	
	filelist = glob.glob(imgdir+'/*.jpeg')
	filelist.sort()
	n = len(filelist)

	imgs1 =[None]*batch_size
	imgs2 =[None]*batch_size

	index1 = range(0,height)
	index2 = range(0,width)

	for x in xrange(0,batch_size):
		imgs = cv2.imread( filelist[ (2*(id+x) )%n ],0 )
		imgs1[x] = imgs[0:height, 0:width][np.newaxis,:,:,np.newaxis]
		
		imgs = cv2.imread( filelist[ (2*(id+x)+1 )%n ],0 )	
		imgs2[x] = imgs[0:height, 0:width][np.newaxis,:,:,np.newaxis]

	return (np.concatenate(imgs1,0), np.concatenate(imgs2,0))


def getGT(output_data, id, batch_size):
	return( output_data[ id:(id+batch_size),:] )



import tensorflow as tf
import numpy as np

def get_warp(p):
	warp = tf.convert_to_tensor( [p[:,0,:]+1, p[:,2,:], p[:,4,:], p[:,1,:], p[:,3,:]+1, p[:,5,:]] )
	warp = tf.transpose( warp, (1,0,2) )
	return warp
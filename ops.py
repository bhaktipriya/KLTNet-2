import tensorflow as tf
import numpy as np
from transformer2 import spatial_transformer_network

def get_warp(p):
	print('afdasdf',p.shape)
	warp = tf.convert_to_tensor( [p[:,0,:]+1, p[:,2,:], p[:,4,:], p[:,1,:], p[:,3,:]+1, p[:,5,:]] )
	warp = tf.transpose( warp, (1,0,2) )
	return warp

def loss(p, p_gt, batch_size, height, width):
	x = tf.ones( (batch_size, height, width,1), tf.float32 )
	
	x_tr = spatial_transformer_network( x, get_warp(p) )
	x_gt = spatial_transformer_network( x, get_warp(p_gt) )

	x_tr = tf.cast( x_tr > 0.6, tf.bool)
	x_gt = tf.cast( x_gt > 0.6, tf.bool)

	intersection = tf.reduce_sum( tf.cast( tf.logical_and(x_gt, x_tr), tf.float32) )
	union = tf.reduce_sum( tf.cast( tf.logical_or(x_gt, x_tr), tf.float32) )

	return( intersection/union )
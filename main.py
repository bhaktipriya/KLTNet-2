import tensorflow as tf
from model_simple import KLTNet
from transformer2 import spatial_transformer_network, affine_grid_generator
from spatial_transformer import transformer, batch_transformer
from klt import *
import numpy as np
import math
import pprint
import cv2


BATCH_SIZE = 1
HEIGHT = 360
WIDTH = 480
CHANNELS = 1


# Define two feature maps
# fmA = tf.ones((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS), dtype=tf.float32)
# fmA = tf.convert_to_tensor(np.random.randint(200, size=(BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)), dtype=tf.float32)

img1 = cv2.imread( 'data/image1.jpg',0)
img2 = cv2.imread( 'data/image2.jpg',0)

# img = img2.astype('float32')
# kerX = np.float32([[-0.5,0.0,0.5]])
# kerY = np.float32([-0.5,0.0,0.5])
# gradX = cv2.filter2D(img,-1,kerX)
# gradY = cv2.filter2D(img,-1,kerY)
# print(gradX)
# print(gradY)

f1 = tf.convert_to_tensor( img1, tf.float32 )
f2 = tf.convert_to_tensor( img2, tf.float32 )

f1 = tf.expand_dims( f1, axis=0)
f1 = tf.expand_dims( f1, axis=3)
f2 = tf.expand_dims( f2, axis=0)
f2 = tf.expand_dims( f2, axis=3)

unity = np.zeros((BATCH_SIZE,2,3), np.float32)
unity[:,0,0] = 1
unity[:,1,1] = 1
unity[:,0,2] = -2
theta = tf.convert_to_tensor( unity )

# theta = tf.convert_to_tensor( np.random.rand( BATCH_SIZE, 2, 3 ), dtype=tf.float32)

# fmB = spatial_transformer_network( fmA, theta )
f3 = spatial_transformer_network( f1, theta )

pp = pprint.PrettyPrinter(indent=4)
arg = {}

net = KLTNet( arg )

net( f1, f3 )

# initialWarp = np.float32([[0.5,0,0],[0,0.5,0]])
# H_inv = lkAffine( img2, img1, initialWarp, 0.01)
# H_inv = cv2.warpAffine(img2, cv2.invertAffineTransform(initialWarp), (HEIGHT, WIDTH) )
# cv2.imwrite( 'temp.png', H_inv )

# init_affine = tf.convert_to_tensor( [2,0,0,0,2,0], tf.float32 )
# init_affine = tf.expand_dims( init_affine, axis=0 )
# init_affine = tf.tile( init_affine, tf.stack([1, 1, 1]) )

# I_warped = spatial_transformer_network( f2, init_affine )

init = tf.global_variables_initializer()

with tf.Session() as sess:
	
	sess.run(init)
	
	writer = tf.summary.FileWriter("/tmp/log/KLTNet", sess.graph)

	# x = sess.run( fmB )
	# j = sess.run( net.J )
	# h = sess.run( net.Hess )
	# h_inv = sess.run( net.H_inv )
	# err = sess.run( net.dp )
	
	# pp.pprint( sess.run( net.H_inv )) 
	pp.pprint( sess.run( net.p ))
	pp.pprint( sess.run( net.dp ))
	# pp.pprint( sess.run( tf.squeeze(net.H_inv) ))
	# pp.pprint( sess.run( tf.squeeze(net.errorImage) ))
	# pp.pprint( sess.run( tf.squeeze(net.diff) ))
	
	# print tf.squeeze(net.diff).get_shape()
	# pp.pprint( sess.run( net.H_list ))
	# np.savetxt( 'x.out', sess.run( tf.squeeze(net.grad_Txs) ) - gradX)
	# np.savetxt( 'y.out', sess.run( tf.squeeze(net.grad_Tys) ) - gradY)

	# np.savetxt( 'H_inv.txt', sess.run( tf.squeeze(net.I_warped) ) - H_inv )
	# imgr = sess.run( tf.squeeze(I_warped) )
	# cv2.imwrite( 'temp1.png', imgr )
	# print(imgr)

	# pp.pprint(sess.run( affine_grid_generator(HEIGHT, WIDTH, unity) ))

	# np.savetxt( 'H_inv.txt', sess.run( tf.squeeze(I_warped)) - H_inv )
	# np.savetxt( 'H_inv.out', sess.run( tf.squeeze(net.I_warped) ))
	# np.savetxt( 'H_inv.out', sess.run( tf.squeeze(net.I) ))

	# pp.pprint( sess.run( tf.squeeze(net.grad_Tx) ) )
	# pp.pprint( sess.run( tf.squeeze(net.grad_Ty) ) )
	# pp.pprint( H_inv )
	# print( sess.run( net.x_s ))

	# theta = tf.convert_to_tensor( [1.0,0,-50,0,1.0,-50], tf.float32 )
	# theta = tf.reshape( theta, (2,3) )
	# print sess.run( theta )
	# print(err.shape)
	# print(y)
	
	writer.close()




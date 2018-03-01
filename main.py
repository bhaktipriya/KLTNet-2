import tensorflow as tf
from model import KLTNet
from transformer import spatial_transformer_network
import numpy as np
import math
import pprint


BATCH_SIZE = 1
HEIGHT = 90
WIDTH = 120
CHANNELS = 1


# Define two feature maps
# fmA = tf.ones((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS), dtype=tf.float32)
fmA = tf.convert_to_tensor(np.random.randint(200, size=(BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)), dtype=tf.float32)

unity = np.zeros((BATCH_SIZE,2,3), np.float32)
unity[:,0,0] = 1
unity[:,1,1] = 1
theta = tf.convert_to_tensor( unity )

# theta = tf.convert_to_tensor( np.random.rand( BATCH_SIZE, 2, 3 ), dtype=tf.float32)

fmB = spatial_transformer_network( fmA, theta )

pp = pprint.PrettyPrinter(indent=4)
arg = {}

net = KLTNet( arg )
net( fmA, fmB )


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
	# pp.pprint( sess.run( net.p ))
	pp.pprint( sess.run( theta ))
	pp.pprint( sess.run( net.H_list ))
	pp.pprint( sess.run( net.grad_Tx))
	# print( sess.run( net.x_s ))

	# print(err.shape)
	# print(y)
	
	writer.close()




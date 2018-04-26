import tensorflow as tf
from model_simple import KLTNet
from transformer2 import spatial_transformer_network, affine_grid_generator
from spatial_transformer import transformer, batch_transformer
import matplotlib.pyplot as plt
from klt import *
from data import getBatch, getGT
from ops import get_warp, loss
import numpy as np
import math
import pprint
import cv2


BATCH_SIZE = 170
HEIGHT = 50
WIDTH = 90
CHANNELS = 1
ID = 0
EPS = 4

X1 = tf.placeholder(tf.float32, shape=[BATCH_SIZE, HEIGHT, WIDTH, CHANNELS])
X2 = tf.placeholder(tf.float32, shape=[BATCH_SIZE, HEIGHT, WIDTH, CHANNELS])
Y = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 6, 1])

img1, img2 = getBatch('2001',ID, BATCH_SIZE, WIDTH, HEIGHT)
output_data = np.genfromtxt('2001/results.csv', delimiter=',')
output = getGT(output_data, ID, BATCH_SIZE)[:,:,np.newaxis]


arg = {}
net = KLTNet( arg )

p, dp = net( X1, X2 )
warp = get_warp(p)
# print p.shape

# loss_op = tf.reduce_mean( tf.square( X2[:,EPS:(HEIGHT-EPS), EPS:(WIDTH-EPS),:] - spatial_transformer_network(X1, warp)[:,5:67, 5:99,:] ), axis=[1,2,3] )
loss_op = loss( p, Y, BATCH_SIZE, HEIGHT, WIDTH )

# transform_op = spatial_transformer_network(X1, warp)
# loss2 = tf.reduce_mean(dp)

init = tf.global_variables_initializer()

config = tf.ConfigProto(
	device_count = {'GPU': 0}
)

with tf.Session( config=config ) as sess:
	
	sess.run(init)
	
	# print( sess.run(p,feed_dict={X1:img1, X2:img2} ) )
	out = sess.run( loss_op, feed_dict={X1:img1, X2:img2, Y:output} )
	print out
	# hist = np.histogram(out)
	# print out
	# print hist
	# plt.hist(out, bins='auto')  # arguments are passed to np.histogram
	# plt.title("Error histogram")
	# plt.xlabel('MSE between transormed image and ground truth')
	# plt.ylabel('count of images')
	# plt.show()
	# print( sess.run( loss2, feed_dict={X1:img1, X2:img2}) )
	# print( np.squeeze(img2) )
	# print( sess.run( tf.squeeze(transform_op), feed_dict={X1:img1, X2:img2}) )

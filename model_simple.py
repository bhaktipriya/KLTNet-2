import tensorflow as tf
import numpy as np
from transformer import affine_grid_generator, bilinear_sampler, spatial_transformer_network

class KLTNet(object):
	"""docstring for KLTNet"""

	def __init__(self, arg):
		self.arg = arg
	

	def __call__(self, img1, img2, is_train=True):
		
		self.is_train = is_train

		self.shape = img1.get_shape()
		self.B = self.shape[0]
		self.H = self.shape[1]
		self.W = self.shape[2]
		self.C = self.shape[3]
		self.H_list = []

		I = img1
		T = img2

		def getgradX():	

			kernel = np.array( [ [ [[-1]] ,[[0]],[[1]] ], [ [[-2]],[[0]],[[2]] ], [ [[-1]],[[0]],[[1]] ] ] , dtype= np.float32  )
			return( kernel )
			
			with tf.variable_scope('kernel', reuse=tf.AUTO_REUSE) as scope:
				x = tf.get_variable("SobelX", kernel)
			return( x )


		def getgradY():

			kernel = np.array( [ [ [[-1]] ,[[-2]],[[-1]] ], [ [[0]],[[0]],[[0]] ], [ [[1]],[[2]],[[1]] ] ] , dtype= np.float32  )
			
			return(kernel)
			with tf.variable_scope('kernel', reuse=tf.AUTO_REUSE) as scope:
				y = tf.get_variable("SobelY", kernel)
			
			return( y )


		def getID(kernel_size):
			x = np.zeros( [kernel_size, kernel_size], dtype=np.float32)
			x[kernel_size/2, kernel_size/2] = 1.0
			return(x)


		self.unit_affine = tf.convert_to_tensor( [ [1,0,0], [0,1,0] ], tf.float32 )
		self.unit_affine = tf.expand_dims( self.unit_affine, axis=0 )
		self.unit_affine = tf.tile( self.unit_affine, tf.stack([self.B, 1, 1]) )

		self.p = self.unit_affine
		
		
		batch_grids = affine_grid_generator(self.H, self.W, self.unit_affine)
		# batch_grids = affine_grid_generator_img(self.H, self.W, self.unit_affine)
		
		x_s = (batch_grids[:, 0, :, :] + 1)*0.5*tf.cast(self.W, 'float32') 		
		x_s = tf.expand_dims( x_s, axis=3 )
		x_s = tf.layers.conv2d(x_s, filters=1, kernel_size=[3,3], kernel_initializer=tf.constant_initializer( getID(3) ),
			 padding="VALID" ,name="id", reuse=tf.AUTO_REUSE, trainable=False)
	
		y_s = (batch_grids[:, 1, :, :] + 1)*0.5*tf.cast(self.H, 'float32')
		y_s = tf.expand_dims( y_s, axis=3 )
		y_s = tf.layers.conv2d(y_s, filters=1, kernel_size=[3,3], kernel_initializer=tf.constant_initializer( getID(3) ),
			 padding="VALID" ,name="id", reuse=tf.AUTO_REUSE, trainable=False)
		

		self.x_s = x_s
		self.y_s = y_s


		for steps in xrange(10):
			
			print(steps)
			## Computing warping image		
			I_warped = spatial_transformer_network(I, self.p)

			## Computing gradient over warped image
			self.grad_Ix = tf.layers.conv2d( I, filters=1, kernel_size=[3,3], kernel_initializer= tf.constant_initializer( getgradX() ), 
				padding="VALID", name="grad_Ix", reuse=tf.AUTO_REUSE, trainable=False)
			
			self.grad_Iy = tf.layers.conv2d( I, filters=1, kernel_size=[3,3], kernel_initializer= tf.constant_initializer( getgradY() ), 
				padding="VALID", name="grad_Iy", reuse=tf.AUTO_REUSE, trainable=False)

			self.grad_Ix = spatial_transformer_network(self.grad_Ix, self.p)
			self.grad_Iy = spatial_transformer_network(self.grad_Iy, self.p)

			## Computing Jacobian
			self.J = tf.convert_to_tensor( [ x_s*self.grad_Ix, x_s*self.grad_Iy, y_s*self.grad_Ix, y_s*self.grad_Iy, self.grad_Ix, self.grad_Iy ] )
			self.J = tf.transpose( self.J, [1,0,2,3,4] )


			## Computing Hessian
			H_list = []
			for i in xrange(6):
				for j in xrange(6):
					H_list.append( self.J[:,i,:,:,:]*self.J[:,j,:,:,:] )

			
			self.H = tf.convert_to_tensor( H_list )
			self.H = tf.reduce_sum( self.H, (2,3,4) )
			self.H = tf.transpose( self.H, [1,0] )
			self.H = tf.reshape( self.H, ( self.B, 6, 6) )
			# self.H_list.append( self.H )
			
			## Computing inverse of Hessian
			self.H_inv = tf.matrix_inverse( self.H )

			## Computing error image
			self.diff = tf.layers.conv2d(T - I_warped, filters=1, kernel_size=[3,3], kernel_initializer=tf.constant_initializer( getID(3) ),
			 padding="VALID" ,name="id", reuse=tf.AUTO_REUSE, trainable=False)
			self.errorImage = self.J*( tf.expand_dims(self.diff , 1) )
			self.errorImage = tf.reduce_sum( self.errorImage, (2,3) )
			
			## Calculating delta p
			self.dp = tf.matmul( self.H_inv, self.errorImage )

			## Updating p
			self.p = self.p + tf.reshape( self.dp, self.p.get_shape() )
			self.H_list.append( tf.reshape( self.p, (self.B, 1, 6) )  )









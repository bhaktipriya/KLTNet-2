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

		paddings = tf.constant([[1, 1], [1,1], [1,1], [1,1]])
		# I = tf.pad(img1, paddings, "SYMMETRIC")
		# T = tf.pad(img2, paddings, "SYMMETRIC")
		I = img1
		T = img2

		def getgradX():	

			# kernel = np.array( [ [ [[-1]] ,[[0]],[[1]] ], [ [[-2]],[[0]],[[2]] ], [ [[-1]],[[0]],[[1]] ] ] , dtype= np.float32  )
			kernel = np.array( [ [ [[0]] ,[[0]],[[0]] ], [ [[-0.5]],[[0]],[[0.5]] ], [ [[0]],[[0]],[[0]] ] ] , dtype= np.float32  )
			return( kernel )
			
			with tf.variable_scope('kernel', reuse=tf.AUTO_REUSE) as scope:
				x = tf.get_variable("SobelX", kernel)
			return( x )


		def getgradY():

			kernel = np.array( [ [ [[0]] ,[[-0.5]],[[0]] ], [ [[0]],[[0]],[[0]] ], [ [[0]],[[0.5]],[[0]] ] ] , dtype= np.float32  )
			
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

		self.init_affine = tf.convert_to_tensor( [[1.0,0.0,10.0/tf.cast(self.W, 'float32')],[0.0,1.0,10.0/tf.cast(self.H, 'float32')]], tf.float32 )
		self.init_affine = tf.expand_dims( self.init_affine, axis=0 )
		self.init_affine = tf.tile( self.init_affine, tf.stack([self.B, 1, 1]) )


		self.p = tf.convert_to_tensor( np.zeros([self.B, 6, 1]), tf.float32 ) 
		
		
		batch_grids = affine_grid_generator(self.H, self.W, self.unit_affine)
		
		x_s = (batch_grids[:, 0, :, :] + 1)*0.5*tf.cast(self.W, 'float32') 		
		x_s = tf.expand_dims( x_s, axis=3 )
		self.x_s = x_s
		x_s = tf.layers.conv2d(x_s, filters=1, kernel_size=[3,3], kernel_initializer=tf.constant_initializer( getID(3) ),
			 padding="VALID" ,name="id", reuse=tf.AUTO_REUSE, trainable=False)
	
		y_s = (batch_grids[:, 1, :, :] + 1)*0.5*tf.cast(self.H, 'float32')
		y_s = tf.expand_dims( y_s, axis=3 )
		y_s = tf.layers.conv2d(y_s, filters=1, kernel_size=[3,3], kernel_initializer=tf.constant_initializer( getID(3) ),
			 padding="VALID" ,name="id", reuse=tf.AUTO_REUSE, trainable=False)
		

		T_padded = tf.pad( T, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
		self.grad_Txs = tf.layers.conv2d( T_padded, filters=1, kernel_size=[3,3], kernel_initializer= tf.constant_initializer( getgradX() ), 
				padding="VALID", name="grad_Tx", reuse=tf.AUTO_REUSE, trainable=False)
		self.grad_Tys = tf.layers.conv2d( T_padded, filters=1, kernel_size=[3,3], kernel_initializer= tf.constant_initializer( getgradY() ), 
				padding="VALID", name="grad_Ty", reuse=tf.AUTO_REUSE, trainable=False)



		self.grad_Tx = tf.layers.conv2d( T, filters=1, kernel_size=[3,3], kernel_initializer= tf.constant_initializer( getgradX() ), 
				padding="VALID", name="grad_Tx", reuse=tf.AUTO_REUSE, trainable=False)
		self.grad_Ty = tf.layers.conv2d( T, filters=1, kernel_size=[3,3], kernel_initializer= tf.constant_initializer( getgradY() ), 
				padding="VALID", name="grad_Ty", reuse=tf.AUTO_REUSE, trainable=False)


		self.J = tf.convert_to_tensor( [ x_s*self.grad_Tx, x_s*self.grad_Ty, y_s*self.grad_Tx, y_s*self.grad_Ty, self.grad_Tx, self.grad_Ty ] )
		self.J = tf.transpose( self.J, (1,0,2,3,4))
		print( self.J.get_shape() )

		H_list = []
		for i in xrange(6):
			for j in xrange(6):
				H_list.append( self.J[:,i,:,:,:]*self.J[:,j,:,:,:] )

		self.Hess = tf.convert_to_tensor( H_list )
		self.Hess = tf.reduce_sum( self.Hess, (2,3,4) )
		self.Hess = tf.transpose( self.Hess, [1,0] )
		self.Hess = tf.reshape( self.Hess, ( self.B, 6, 6) )
		print( self.Hess.get_shape() )

		self.H_inv = tf.matrix_inverse( self.Hess )
		self.H_list.append( self.H_inv )
		# print( self.Hess.get_shape() )

		for steps in xrange(1):
			
			# print(steps)
			## Computing warping image
			self.warp = tf.convert_to_tensor( [self.p[:,0,:]+1, self.p[:,2,:], self.p[:,4,:], self.p[:,1,:], self.p[:,3,:]+1, self.p[:,5,:]] )
			self.warp = tf.transpose( self.warp, (1,0,2) )		
			# print( self.warp.get_shape() )
			
			I_warped = spatial_transformer_network(I, self.warp)
			self.I_warped = spatial_transformer_network( I, self.init_affine )
			self.I = I

			## Computing error image
			self.diff = tf.layers.conv2d( I_warped - T, filters=1, kernel_size=[3,3], kernel_initializer=tf.constant_initializer( getID(3) ),
			 padding="VALID" ,name="id", reuse=tf.AUTO_REUSE, trainable=False)
		

			self.errorImage = self.J*( tf.expand_dims(self.diff , 1) )
			self.errorImage = tf.reduce_sum( self.errorImage, (2,3) )

			## Calculating delta p
			self.dp = tf.matmul( self.H_inv, self.errorImage )

			## Updating p
			sum_p = self.p + self.dp
			temp = [ sum_p[:,0,:] + self.p[:,0,:]*self.dp[:,0,:] + self.p[:,2,:]*self.dp[:,1,:],
			sum_p[:,1,:] + self.p[:,1,:]*self.dp[:,0,:] + self.p[:,3,:]*self.dp[:,1,:],
			sum_p[:,2,:] + self.p[:,0,:]*self.dp[:,2,:] + self.p[:,2,:]*self.dp[:,3,:],
			sum_p[:,3,:] + self.p[:,1,:]*self.dp[:,2,:] + self.p[:,3,:]*self.dp[:,3,:],
			sum_p[:,4,:] + self.p[:,0,:]*self.dp[:,4,:] + self.p[:,2,:]*self.dp[:,5,:],
			sum_p[:,5,:] + self.p[:,1,:]*self.dp[:,4,:] + self.p[:,3,:]*self.dp[:,5,:]]
			
			self.p = tf.convert_to_tensor( temp )
			self.p = tf.transpose( self.p, (1,0,2) )
			# print( self.p.get_shape() )








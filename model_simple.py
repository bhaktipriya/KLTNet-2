import tensorflow as tf
import numpy as np
from transformer2 import affine_grid_generator, bilinear_sampler, spatial_transformer_network
from ops import get_warp

class KLTNet(object):
	"""docstring for KLTNet"""

	def __init__(self, arg):
		self.arg = arg
	

	def __call__(self, img1, img2, is_train=True):
		
		self.is_train = is_train

		self.shape = img1.get_shape().as_list()
		self.B = self.shape[0]
		self.H = self.shape[1]
		self.W = self.shape[2]
		self.C = self.shape[3]
		self.H_list = []

		paddings = tf.constant( [[0,0],[1,1],[1,1],[0,0]] )

		I = tf.pad(img1, paddings, "REFLECT")
		T = img2
		# T = tf.pad(img2, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")

		def getgradX():	
			# kernel = np.array( [ [ [[-1]] ,[[0]],[[1]] ], [ [[-2]],[[0]],[[2]] ], [ [[-1]],[[0]],[[1]] ] ] , dtype= np.float32  )
			kernel = np.array( [ [ [[0]] ,[[0]],[[0]] ], [ [[-0.5]],[[0]],[[0.5]] ], [ [[0]],[[0]],[[0]] ] ] , dtype= np.float32  )
			return( kernel )


		def getgradY():
			kernel = np.array( [ [ [[0]] ,[[-0.5]],[[0]] ], [ [[0]],[[0]],[[0]] ], [ [[0]],[[0.5]],[[0]] ] ] , dtype= np.float32  )	
			return(kernel)


		def getID(kernel_size):
			x = np.zeros( [kernel_size, kernel_size], dtype=np.float32)
			x[kernel_size/2, kernel_size/2] = 1.0
			return(x)

		def removeSides(x, kernel_size=3):
			with tf.variable_scope('kernel', reuse=tf.AUTO_REUSE) as scope:
				x = tf.layers.conv2d(x, filters=1, kernel_size=[kernel_size,kernel_size], kernel_initializer=tf.constant_initializer( getID(kernel_size) ),
			 padding="VALID" ,name="id", reuse=tf.AUTO_REUSE, trainable=False)
			return(x)



		self.unit_affine = tf.convert_to_tensor( [ [1,0,0], [0,1,0] ], tf.float32 )
		self.unit_affine = tf.expand_dims( self.unit_affine, axis=0 )
		self.unit_affine = tf.tile( self.unit_affine, tf.stack([self.B, 1, 1]) )

		
		batch_grids = affine_grid_generator(self.H, self.W, self.unit_affine)
		# batch_grids = affine_grid_generator_img(self.H, self.W, self.unit_affine)
		
		x_s = batch_grids[:, 0, :, :] 		
		x_s = tf.expand_dims( x_s, axis=3 )
		x_s = removeSides(x_s)
	
		y_s = batch_grids[:, 1, :, :]
		y_s = tf.expand_dims( y_s, axis=3 )
		y_s = removeSides(y_s)
		

		self.x_s = x_s
		self.y_s = y_s

		## Initializing parameters with zeros
		# unity = np.zeros( (self.B,6,1), np.float32 )
		# unity[:,0,:] = 1.0
		# unity[:,4,:] = 1.0
		# self.p = tf.constant( unity )
		self.p = tf.zeros( (self.B,6,1), tf.float32 )

		for steps in xrange(10):
			
			print(steps)
			## Computing warping image	
			# self.warp = tf.convert_to_tensor( [self.p[:,0,:]+1, self.p[:,2,:], self.p[:,4,:], self.p[:,1,:], self.p[:,3,:]+1, self.p[:,5,:]] )
			# self.warp = tf.transpose( self.warp, (1,0,2) )
			self.warp = get_warp( self.p )		
			
			I_warped = spatial_transformer_network(I, self.warp)
			I_warped = removeSides(I_warped)
			
			## Computing gradient over warped image
			self.grad_Ix = tf.layers.conv2d( I, filters=1, kernel_size=[3,3], kernel_initializer= tf.constant_initializer( getgradX() ), 
				padding="VALID", name="grad_Ix", reuse=tf.AUTO_REUSE, trainable=False)
			
			self.grad_Iy = tf.layers.conv2d( I, filters=1, kernel_size=[3,3], kernel_initializer= tf.constant_initializer( getgradY() ), 
				padding="VALID", name="grad_Iy", reuse=tf.AUTO_REUSE, trainable=False)

			
			## Warping the gradient
			self.grad_Ix = spatial_transformer_network(self.grad_Ix, self.warp)
			self.grad_Ix = removeSides(self.grad_Ix)
			
			self.grad_Iy = spatial_transformer_network(self.grad_Iy, self.warp)
			self.grad_Iy = removeSides(self.grad_Iy)
			

			## Computing Jacobian
			self.J = tf.convert_to_tensor( [ x_s*self.grad_Ix, x_s*self.grad_Iy, y_s*self.grad_Ix, y_s*self.grad_Iy, self.grad_Ix, self.grad_Iy ] )
			self.J = tf.transpose( self.J, [1,0,2,3,4] )

			## Computing Hessian
			H_list = []
			for i in xrange(6):
				for j in xrange(6):
					H_list.append( self.J[:,i,:,:,:]*self.J[:,j,:,:,:] )

			
			self.Hess = tf.convert_to_tensor( H_list )
			self.Hess = tf.reduce_sum( self.Hess, (2,3,4) )
			self.Hess = tf.transpose( self.Hess, [1,0] )
			self.Hess = tf.reshape( self.Hess, ( self.B, 6, 6) )
			
			## Computing inverse of Hessian
			self.H_inv = tf.matrix_inverse( self.Hess )

			## Computing error image
			self.diff = removeSides( T - I_warped)			
			self.errorImage = self.J*( tf.expand_dims( self.diff , 1) )
			self.errorImage = tf.reduce_sum( self.errorImage, (2,3) )
			
			## Calculating delta p
			self.dp = tf.matmul( self.H_inv, self.errorImage )

			## Updating p
			self.p = self.p + self.dp

		return (self.p, self.dp)
			








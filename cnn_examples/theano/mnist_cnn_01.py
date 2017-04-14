
## THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=1'  python <myscript>.py
## THEANO_FLAGS='floatX=float32,device=gpu0'  python <myscript>.py
## THEANO_FLAGS='floatX=float32'  python <myscript>.py

##from __future__ import print_function
import timeit

import numpy 
import gzip
import sys
import os, struct

from pylab import *
from numpy import *

from array import array as pyarray
from numpy import append, array, int8, uint8, zeros

######## 

import theano
import theano.tensor as T
import six.moves.cPickle as pickle

from theano.tensor.signal import pool

def load_data():
	############ testing data #########################################################

	f_testing_img = gzip.open('t10k-images-idx3-ubyte.gz','rb')

	####  In big endian, you store the most significant byte in the smallest address
	####  >	---> big-endian
	####  I ---> unsigned int

	magic_nr, testing_size, rows, cols = struct.unpack(">IIII", f_testing_img.read(16))
	print "testing_size = %d" % (testing_size)

	#### B --> unsigned char
	testing_img = pyarray("B", f_testing_img.read())
        testing_img = array(testing_img).reshape((10000, 28*28)) #### !!!!!

	f_testing_img.close()

	f_testing_lbl = gzip.open('t10k-labels-idx1-ubyte.gz','rb')
	magic_nr, testing_size = struct.unpack(">II", f_testing_lbl.read(8) )

	##...  ... [0] 0
	##...  ... [1] 0
	##...  ... [2] 8
	##...  ... [3] 1
	##...  ... [4] 0
	##...  ... [5] 0
	##...  ... [6] 39
	##...  ... [7] 16
	##........................................
	##...  ... [8] 7
	##...  ... [9] 2
	##...  ... [10] 1
	##...  ... [11] 0
	##...  ... [12] 4
	##...  ... [13] 1
	##...  ... [14] 4
	##...  ... [15] 9
	##...  ... [16] 5
	##...  ... [17] 9
	##...  ... [18] 0
	##...  ... [19] 6

	#### b -> signed char
	testing_lbl = pyarray("b", f_testing_lbl.read())
	f_testing_lbl.close()

	print "testing_size = %d" % (testing_size)

	for k in range(20):
		print "... labels ... [%d] %d" % (k, testing_lbl[k])

	##... labels ... [0] 7
	##... labels ... [1] 2
	##... labels ... [2] 1
	##... labels ... [3] 0
	##... labels ... [4] 4
	##... labels ... [5] 1
	##... labels ... [6] 4
	##... labels ... [7] 9
	##... labels ... [8] 5
	##... labels ... [9] 9
	##... labels ... [10] 0
	##... labels ... [11] 6
	##... labels ... [12] 9
	##... labels ... [13] 0
	##... labels ... [14] 1
	##... labels ... [15] 5
	##... labels ... [16] 9
	##... labels ... [17] 7
	##... labels ... [18] 3
	##... labels ... [19] 4

	##################### training data #######################################################

	f_training_img = gzip.open('train-images-idx3-ubyte.gz','rb')

	magic_nr, training_size, rows, cols = struct.unpack(">IIII", f_training_img.read(16))
	training_img = pyarray("B", f_training_img.read())
	print "len(training_img) = ", len(training_img)
	print "len(training_img)/(rows*cols) = %d" % ( len(training_img)/(rows*cols) )
	f_training_img.close()
	print ".. training_size = %d" % ( training_size )
	print ".. rows = %d" % (rows)
	print ".. cols = %d" % (cols)

        
     

	ml_training_img = training_img[ 0: 50000*rows*cols ]
	print ".. len(ml_training_img) = %d" % len( ml_training_img )
	print ".. len(ml_training_img)/(rows*cols) = %f" % (len( ml_training_img )/float(rows*cols))

        ml_training_img = array(ml_training_img).reshape((50000, 28*28)) #### !!!!!

	ml_valid_img = training_img[50000*rows*cols: ]
	print ".. len(ml_valid_img) = %d" % len( ml_valid_img )
	print ".. len(ml_valid_img)/(rows*cols) = %f" % (len( ml_valid_img )/float(rows*cols))

        ml_valid_img = array(ml_valid_img).reshape((10000, 28*28)) #### !!!!!

	############ labels ########

	f_training_lbl = gzip.open('train-labels-idx1-ubyte.gz','rb')
	magic_nr, training_size = struct.unpack(">II", f_training_lbl.read(8) )
	training_lbl = pyarray("b", f_training_lbl.read())
	f_training_lbl.close()

        print ".. len(traing_lbl) = %d" % (len(training_lbl))    
 
	ml_training_label = training_lbl[0: 50000 ]
	ml_valid_label = training_lbl[50000: ]


	print ""
	print ".. training_size = %d" % (training_size)
	print ""

	#################################################################################################

	display = False

	if(display):
	
		ind = [ k for k in range(testing_size) if testing_lbl[k] in [2] ]
		N = len(ind)

		images = zeros((N, rows, cols), dtype=uint8)
		labels = zeros((N, 1), dtype=int8)

		for i in range(len(ind)):
  			images[i] = array(testing_img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        		labels[i] = testing_lbl[ind[i]]
	
		#########################################################################################
	
		imshow(images.mean(axis=0), cmap=cm.gray)
		show()
	
	
	##################################################################################################

	######### tuples ############
	train_set = (ml_training_img, ml_training_label)
	valid_set = (ml_valid_img, ml_valid_label)
	test_set = (testing_img, testing_lbl)


	def shared_dataset(data_x_y, borrow = True):
                
		data_x, data_y = data_x_y
		shared_x = theano.shared(value = numpy.asarray(data_x, dtype = theano.config.floatX), borrow = borrow)
        	shared_y = theano.shared(value = numpy.asarray(data_y, dtype = theano.config.floatX), borrow = borrow)
		
		return ( shared_x, T.cast(shared_y, 'int32') )
	
	
	train_set_x, train_set_y = shared_dataset(train_set)
	valid_set_x, valid_set_y = shared_dataset(valid_set)
	test_set_x, test_set_y   = shared_dataset(test_set)
	
	###### return value
	
	rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
	
    	return rval
	




### n_rows // batch_size



###############################




##
## input
##
##               | row 1 (n_hidden_layer) 
##               | row 2 (n_hidden_layer)
##               | '
##               | '
## batch_size ---| 
##		 |	
##               |
##               | '
##               | row n (n_hidden_layer)
##
##

##
## W
##                     		 | row 1 (10) (number 0 ~ number 9)
##                     		 | row 2 (10)
##                     		 |             
## 28-by-28 (n_hidden_layer) ----|
##                     		 | 
##                     		 |
##                     		 | row n (10)
##

##
##
## P(Y=i | x, W, b ) = softmax(Wx +b)
##
##		        exp(W_i*x + b_i)
##		=     ------------------------------			
##			sum_j ( exp(W_j*x + g_j) )	
##


class LogisticRegression(object):
	
	def __init__(self, input, n_in, n_out):

		self.W = theano.shared( value = numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
					name ='W',
					borrow = True
				      )

		self.b = theano.shared( value = numpy.zeros((n_out,), dtype = theano.config.floatX),
					name = 'b',
					borrow = True
				      )					 

		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b ) ## self.b (row) is broadcasted
		

                ### theano.tensor.argmax :  the index of the maximum value along a given axis
		self.y_pred = T.argmax( self.p_y_given_x, axis = 1) ### axis=1 ---> row direction
		
		self.params = [self.W, self.b]
		
		self.input = input
		
	def negative_log_likelihood(self, y):  #### cost
		
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

	def errors(self, y):

		if y.ndim != self.y_pred.ndim:
            		raise TypeError(
                		'y should have the same shape as self.y_pred',
                		('y', y.type, 'y_pred', self.y_pred.type)
           		)

		if y.dtype.startswith('int'):
			
			return T.mean(T.neq(self.y_pred, y))

		else:
			raise NotImplementedError()


class HiddenLayer(object):

    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        
        self.input = input  ###  T.matrix('x')
        
        if W is None:
		
            W_values = numpy.asarray( rng.uniform(
                    					low=-numpy.sqrt(6. / (n_in + n_out)),
                    					high=numpy.sqrt(6. / (n_in + n_out)),
                    					size=(n_in, n_out)
                		     ),
                		     dtype=theano.config.floatX
            )
    		
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        
        self.params = [self.W, self.b]


### input ---> (batch_size, height, width) ----> ( batch_size, n_filters, height, width)
### filter_shape ---> ( n_out_filters, n_input_filters, height, width)      

class LeNetConvPoolLayer(object):
    
    
    def __init__( self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        
        
        assert image_shape[1] == filter_shape[1]
        self.input = input
        
        fan_in = numpy.prod( filter_shape[1:] )
        
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) // numpy.prod(poolsize))
        
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared( numpy.asarray(
                                               rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                                               dtype=theano.config.floatX
                                             ),
                                             borrow=True
        )
	
        
        b_values = numpy.zeros( (filter_shape[0], ), dtype=theano.config.floatX)
        self.b = theano.shared( value=b_values, borrow=True)
        
        
        conv_out = T.nnet.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        ###
	### ignore_border (bool (default None, will print a warning and set to False))
        ###  When True, (5,5) input with ds=(2,2) will generate a (2,2) output. (3,3) otherwise.

        ###  
	###  0 1 2 3 4           0 1    2 3   4 x (border)
	###  1 2 3 4 5  --->     1 2    3 4   5 x (border)
	###  2 3 4 5 6        
	###  3 4 5 6 7           2 3    4 5   6 x (border)
	###  4 5 6 7 8           3 4    5 6   7 x (border)
	###
	###                      4 5    6 7   8 x (border)
        ###                      x x    x x   x x (border)
        ###                   (border) (border)
        ###                       
	

	###  from theano.tensor.signal import pool

	### input = T.dtensor4('input')
	### maxpool_shape = (2, 2)
	### pool_out = pool.pool_2d(input, maxpool_shape, ignore_border=True)
	### f = theano.function([input],pool_out)

	### invals = numpy.random.RandomState(1).rand(3, 2, 5, 5)
	### print 'With ignore_border set to True:'
	### print 'invals[0, 0, :, :] =\n', invals[0, 0, :, :]
	### print 'output[0, 0, :, :] =\n', f(invals)[0, 0, :, :]

	### pool_out = pool.pool_2d(input, maxpool_shape, ignore_border=False)
	### f = theano.function([input],pool_out)
	### print 'With ignore_border set to False:'
	### print 'invals[1, 0, :, :] =\n ', invals[1, 0, :, :]
	### print 'output[1, 0, :, :] =\n ', f(invals)[1, 0, :, :]
	###
	
	### With ignore_border set to True:
    	### invals[0, 0, :, :] =
    	### [[  4.17022005e-01   7.20324493e-01   1.14374817e-04   3.02332573e-01 1.46755891e-01]
     	### [  9.23385948e-02   1.86260211e-01   3.45560727e-01   3.96767474e-01 5.38816734e-01]
     	### [  4.19194514e-01   6.85219500e-01   2.04452250e-01   8.78117436e-01 2.73875932e-02]
     	### [  6.70467510e-01   4.17304802e-01   5.58689828e-01   1.40386939e-01 1.98101489e-01]
     	### [  8.00744569e-01   9.68261576e-01   3.13424178e-01   6.92322616e-01 8.76389152e-01]]
    	### output[0, 0, :, :] =
    	### [[ 0.72032449  0.39676747]
     	### [ 0.6852195   0.87811744]]

	###
	### With ignore_border set to False:
    	### invals[1, 0, :, :] =
    	### [[ 0.01936696  0.67883553  0.21162812  0.26554666  0.49157316]
     	### [ 0.05336255  0.57411761  0.14672857  0.58930554  0.69975836]
     	### [ 0.10233443  0.41405599  0.69440016  0.41417927  0.04995346]
     	### [ 0.53589641  0.66379465  0.51488911  0.94459476  0.58655504]
     	### [ 0.90340192  0.1374747   0.13927635  0.80739129  0.39767684]]
    	### output[1, 0, :, :] =
    	### [[ 0.67883553  0.58930554  0.69975836]
     	### [ 0.66379465  0.94459476  0.58655504]
     	### [ 0.90340192  0.80739129  0.39767684]]
	###
	
	### theano.tensor.signal.pool.pool_2d	

        ##pooled_out = T.signal.pool.pool_2d(

        pooled_out = pool.pool_2d(
        
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        
        self.params = [self.W, self.b]
        
        self.input = input
	

def evaluate_lenet5(learning_rate=0.1, n_epochs=2, nkerns=[20, 50], batch_size = 500):
    
    rng = numpy.random.RandomState(23455)
    
    datasets = load_data()
    
    train_set_x, train_set_y = datasets[0]          ### = (train_set_x, train_set_y )
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y   = datasets[2]

    print "type(test_set_y)= ", type(test_set_y)
    print "test_set_y.type()= ", test_set_y.type()
    print ""
    
    

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches  = test_set_x.get_value(borrow=True).shape[0]

    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size
    
    print ""
    print "   batch_size = %d " % (batch_size)
    print "   n_train_batches = %d " % (n_train_batches)
    print "   n_valid_batches = %d " % (n_valid_batches)
    print "   n_test_batches  = %d " % (n_test_batches)
    print ""
    

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
        
    print('... building the model, so wait for a while')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28) to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape( (batch_size, 1, 28, 28) )
    
    
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape  =  (batch_size, 1, 28, 28),
        filter_shape =  (nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    
    
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape   = (batch_size, nkerns[0], 12, 12),
        filter_shape  = (nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )
    
    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    
    layer2_input = layer1.output.flatten(2)
    
    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=500,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model

    temp_x = test_set_x[index * batch_size: (index + 1) * batch_size]

    print ""
    print "type(temp_x) = ", type(temp_x)
    print "temp_x.type() = ", temp_x.type()
    print ""

    ### temp_x.type() =  <TensorType(float64, vector)>
    ### Cannot convert Type TensorType(float64, vector) into Type TensorType(float64, matrix)

    temp_y = test_set_y[index * batch_size: (index + 1) * batch_size]
    print ""
    print "type(temp_y) = ", type(temp_y)
    print "temp_y.type() = ", temp_y.type()
    print ""

    test_model = theano.function(
        [index],
        layer3.errors(y),                          
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),                         
        givens = {
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params
    
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    
    # train_model is a function that updates the model parameters by SGD Since this model has many parameters, 
    # it would be tedious to manually create an update rule for each model parameter. 
    # We thus create the updates list by automatically looping over all (params[i], grads[i]) pairs.
    
    updates = [
        (param_i, param_i - learning_rate * grad_i)  for param_i, grad_i in zip(params, grads)
    ]
    
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens = {
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
     
    
    print('... training, so wait again')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found

    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many minibatche before checking the network
                                  # on the validation set; in this case we check every epoch
    
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()
     
    epoch = 0
    done_looping = False
    
    while ( epoch < n_epochs ) and ( not done_looping ):
        
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 10 == 0:
                print('training @ iter = ', iter)
            
	    #### < training >	
            cost_ij = train_model(minibatch_index)
            
            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print 'epoch %i, minibatch %i/%i, validation error %f ' % (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.)

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss*improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i) for i in range(n_test_batches)
                    ]
		
                    test_score = numpy.mean(test_losses)
                    print '     epoch %i, minibatch %i/%i, test error of best model %f ' % (epoch, minibatch_index + 1, n_train_batches, test_score * 100.)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print 'Optimization complete.'
    print 'Best validation score of %f %% obtained at iteration %i, with test performance %f' %  (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    
    print 'The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.)

if __name__ == '__main__':
    evaluate_lenet5()

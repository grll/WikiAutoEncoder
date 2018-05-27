import tensorflow as tf
import numpy as np
import time
from .helpers import TimeDiff
rng = np.random

class AutoEncoder:
    """Implements and manage an auto-encoder for dimensionality reduction.
    
    The auto-encoder implemented in this class use a a classic and simple designed Neural Network
    with 3 hidden layers. The first layer has 256 neurons, the second layer (used as encoded layer)
    has 128 neurons, and a third layer to decode the data has again 256 neurons. The activation
    function used here is a simple sigmoid with weights and biases. MSE between inputs and outputs
    is used as cost function.

    Attributes:
        X (tf[placeholder][None, input_size]): single input or batch of inputs for the NN.
        y_pred (tf[operation][None, input_size]): single or multiple output from the NN.
        X_encoded (tf[operation][None, reduced_dims2]): output from the 2 encoding layers.
        loss (tf[operation][1]): Cost function (MSE).
        optimizer (tf[train]): Optimizer function here using RMSPropOptimizer.
        init (tf[operation]): Tensorflow variable initialize function.
        saver (tf[saver]): Tensorflow saver implementation.
        session (tf[session]): Tensorflow session in which all the operations are ran.
    """
    def __init__(self, dim_input, reduced_dim1=256, reduced_dim2=128, learning_rate=0.01):
        """Initialize the NeuralNetwork with the specifed parameters.
        
        Args:
            dim_input (int): Dimension of the input layer.
            reduced_dim1 (int): Dimension of the first reduced layer.
            reduced_dim2 (int): Dimension of the second reduced layer (encoded dimension).
            learning_rate (float): Learning rate used by the optimizer function (0.01 by default).
        """
        self.X = tf.placeholder("float", [None, dim_input])

        weights = {
            'encoder_h1': tf.Variable(tf.random_normal([dim_input, reduced_dim1])),
            'encoder_h2': tf.Variable(tf.random_normal([reduced_dim1, reduced_dim2])),
            'decoder_h1': tf.Variable(tf.random_normal([reduced_dim2, reduced_dim1])),
            'decoder_h2': tf.Variable(tf.random_normal([reduced_dim1, dim_input])),
        }

        biases = {
            'encoder_b1': tf.Variable(tf.random_normal([reduced_dim1])),
            'encoder_b2': tf.Variable(tf.random_normal([reduced_dim2])),
            'decoder_b1': tf.Variable(tf.random_normal([reduced_dim1])),
            'decoder_b2': tf.Variable(tf.random_normal([dim_input])),
        }

        # Building the encoder
        def encoder(x):
            layer_1 = tf.nn.sigmoid(tf.add(
                tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
            layer_2 = tf.nn.sigmoid(tf.add(
                tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
            return layer_2

        # Building the decoder
        def decoder(x):
            layer_1 = tf.nn.sigmoid(tf.add(
                tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
            layer_2 = tf.nn.sigmoid(tf.add(
                tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
            return layer_2
        
        # Construct model
        encoder_op = encoder(self.X)
        decoder_op = decoder(encoder_op)

        # Prediction
        self.y_pred = decoder_op
        # Targets (Labels) are the input data.
        y_true = self.X
        # Encoded
        self.X_encoded = encoder_op

        # Define loss and optimizer, minimize the squared error
        self.loss = tf.reduce_mean(tf.pow(y_true - self.y_pred, 2))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        self.session = tf.Session()

    def trainNN(self, batch_generator_func, batch_size=32, num_steps=20): 
        """Train the neural network previously initialized using mini bacthes of data.
        
        Args:
            batch_generator_func (Func): Function returning a generator of batches with batch_size.
            batch_size (int): The size of the batches to use for training (default 32).
            num_steps (int): The number of steps to use (number of iteration or epoch).
        """
        self.session.run(self.init)

        for i in range(1, num_steps+1):
            print("=== Step %i ===" % i)
            start = time.time()
            losses = []
            for batch_x in batch_generator_func(batch_size):
                _, loss = self.session.run([self.optimizer, self.loss], feed_dict={self.X: batch_x})
                losses.append(loss)
            end = time.time()
            hours, minutes, seconds = TimeDiff(start, end).hms()
            print("Trained in {}h {}min {}s.".format(hours, minutes, seconds))
            print("Last Minibatch loss was: {}".format(loss))
            print("Average loss was: {}".format(np.mean(losses)))
        
        self.save_state()

    def save_state(self, filename=None):
        """Save the state from a previously trained NN.
        
        Args:
            filename (str): filename under which to save the state.
        """
        fname = filename or 'embedding.chk'
        self.saver.save(self.session, './saved_states/tf/' + fname)
        print("Training state successfully saved.")

    def restore_state(self, filename=None):
        """Load the state from a previously trained NN.
        
        Args:
            filename (str): filename under which to load the state.
        """
        fname = filename or 'embedding.chk'
        self.saver.restore(self.session, './saved_states/tf/' + fname)
        print("Training state successfully restored.")

    def create_embedding(self, tokenvec):
        """Generate the encoded version of a given tokenvec from the current state of the NN.
        
        Args:
            tokenvec (array): a token vector or vectors to encode.
        """
        return self.session.run(self.X_encoded, feed_dict={self.X: tokenvec})

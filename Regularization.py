# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

num_steps = 501
num_nn_steps = 501
l2_loss_param = 0.01 #90.0% test accuracy achieved on the nn, 90.3% with dropout
pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	train_dataset = save['train_dataset']
	train_labels = save['train_labels']
	valid_dataset = save['valid_dataset']
	valid_labels = save['valid_labels']
	test_dataset = save['test_dataset']
	test_labels = save['test_labels']
	del save	# hint to help gc free up memory
	print 'Training set', train_dataset.shape, train_labels.shape
	print 'Validation set', valid_dataset.shape, valid_labels.shape
	print 'Test set', test_dataset.shape, test_labels.shape

image_size = 28
num_labels = 10

def reformat(dataset, labels):
	dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
	# Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
	return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print 'Training set', train_dataset.shape, train_labels.shape 
print 'Validation set', valid_dataset.shape, valid_labels.shape 
print 'Test set', test_dataset.shape, test_labels.shape


def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
					/ predictions.shape[0])

# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
train_subset = 10000
batch_size = 128

graph = tf.Graph()
with graph.as_default():

	# Input data. For the training data, we use a placeholder that will be fed
	# at run time with a training minibatch.
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)
	
	# Variables.
	weights = tf.Variable(
		tf.truncated_normal([image_size * image_size, num_labels]))
	biases = tf.Variable(tf.zeros([num_labels]))
	
	# Training computation.
	logits = tf.matmul(tf_train_dataset, weights) + biases
	loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))+tf.nn.l2_loss(weights)*l2_loss_param
	
	# Optimizer.
	optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
	
	# Predictions for the training, validation, and test data.
	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(
		tf.matmul(tf_valid_dataset, weights) + biases)
	test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

with tf.Session(graph=graph) as session:
	tf.initialize_all_variables().run()
	for step in xrange(num_steps):
		# Pick an offset within the training data, which has been randomized.
		# Note: we could use better randomization across epochs.
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		# Generate a minibatch.
		batch_data = train_dataset[offset:(offset + batch_size), :]
		batch_labels = train_labels[offset:(offset + batch_size), :]
		# Prepare a dictionary telling the session where to feed the minibatch.
		# The key of the dictionary is the placeholder node of the graph to be fed,
		# and the value is the numpy array to feed to it.
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
		_, l, predictions = session.run(
			[optimizer, loss, train_prediction], feed_dict=feed_dict)
		if (step % 500 == 0):
			print "Minibatch loss at step", step, ":", l
			print "Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels)
			print "Validation accuracy: %.1f%%" % accuracy(
				valid_prediction.eval(), valid_labels)
	print "Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels)

graph = tf.Graph()
hidden_layer_size = 1024;
with graph.as_default():
	# input data
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)
	#variables
	hidden_weights = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_layer_size]))
	hidden_bias = tf.Variable(tf.zeros([hidden_layer_size]))
	output_weights = tf.Variable(tf.truncated_normal([hidden_layer_size, num_labels]))
	output_bias = tf.Variable(tf.zeros([num_labels]))
	#forward prop
	def forward_prop(dataset,train_flag=True):
		hidden_layer = tf.nn.relu(tf.matmul(dataset,hidden_weights)+hidden_bias)
		if train_flag:
			hidden_layer = tf.nn.dropout(hidden_layer,0.5)
		return tf.matmul(hidden_layer,output_weights)+output_bias
	#loss
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=forward_prop(tf_train_dataset),labels=tf_train_labels))+(tf.nn.l2_loss(hidden_weights)+tf.nn.l2_loss(output_weights))*l2_loss_param
	#optimizer
	optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
	#predictions
	train_prediction = tf.nn.softmax(forward_prop(tf_train_dataset,False))
	valid_prediction = tf.nn.softmax(forward_prop(tf_valid_dataset,False))
	test_prediction = tf.nn.softmax(forward_prop(tf_test_dataset,False))
	
with tf.Session(graph=graph) as session:
	tf.initialize_all_variables().run()
	for step in xrange(num_nn_steps):
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		# Generate a minibatch.
		batch_data = train_dataset[offset:(offset + batch_size), :]
		batch_labels = train_labels[offset:(offset + batch_size), :]
		# Prepare a dictionary telling the session where to feed the minibatch.
		# The key of the dictionary is the placeholder node of the graph to be fed,
		# and the value is the numpy array to feed to it.
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
		_, l, predictions = session.run(
			[optimizer, loss, train_prediction], feed_dict=feed_dict)
		if (step % 500 == 0):
			print "Minibatch loss at step", step, ":", l
			print "Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels)
			print "Validation accuracy: %.1f%%" % accuracy(
				valid_prediction.eval(), valid_labels)
	print "Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels)
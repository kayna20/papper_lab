# View more python learning tutorial on my Youtube and Youku channel!!!

# My tutorial website: https://morvanzhou.github.io/tutorials/

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Import MNIST data
from datasets.kdd99 import input_data
kdd99 = input_data.read_data_sets()

# Visualize encoder setting
# Parameters
learning_rate = 0.01    # 0.01 this learning rate will be better! Tested
training_epochs = 10  #训练次数，把整个训练集过一遍叫做一个epoch，
batch_size = 512    #迭代次数，训练时使用随机梯度法整体计算损失函数的样本数叫做batch_size
display_step = 1

# Network Parameters
n_input = 41  # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

# hidden layer settings
n_hidden_1 = 36
n_hidden_2 = 24 
n_hidden_3 = 16
n_hidden_4 = 10

weights = {
    'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],)),
    'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],)),
    'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],)),
    'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4],)),

    'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3],)),
    'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2],)),
    'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1],)),
    'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_1, n_input],)),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),

    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b4': tf.Variable(tf.random_normal([n_input])),
}


def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    layer_4 = tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                                    biases['encoder_b4'])
    return layer_4


def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                biases['decoder_b4']))
    return layer_4

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
#ADAM通常会取得比较好的结果，同时收敛非常快相比SGD L-BFGS适用于全batch做优化的情况 
#有时候可以多种优化方法同时使用，比如使用SGD进行warm up，然后ADAM 对于比较奇怪的需求，
#deepbit两个loss的收敛需要进行控制的情况，比较慢的SGD比较适用
#SGD
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
#Momentum
#optimizer = tf.train.MomentumOptimizer(lr, 0.9)

#AdaGrad
#optimizer = tf.train.AdagradientOptimizer(learning_rate=self.learning_rate)

#RMSProp
#optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)

#ADAM
#optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-08)

# Launch the graph
with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    total_batch = int(kdd99.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = kdd99.train.random_select(batch_size)  # max(x) = 1, min(x) = 0
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    svm_train_xs,svm_train_ys = kdd99.train.random_select()
    svm_encoder_result = sess.run(encoder_op,feed_dict={X:svm_train_xs})


    tXs,tYs = kdd99.test.random_select(1000)
    colors = [list('rg')[int(x-1)] for x in tYs]
    encoder_result = sess.run(encoder_op, feed_dict={X: tXs})

    clf = svm.SVC()
    clf.fit(svm_encoder_result,svm_train_ys)

    result = clf.predict(encoder_result)
    print(np.mean(np.equal(result,tYs)))
    #plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c = colors)
    #plt.show()


# View more python learning tutorial on my Youtube and Youku channel!!!

# My tutorial website: https://morvanzhou.github.io/tutorials/

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import datetime
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# Import MNIST data
from datasets.kdd99 import input_data

class MyAutoEncoder(object):
    def __init__(self,layers=[64,32,16]):
        #tensorflow session
        self.tf_sess = tf.Session()
        # Visualize encoder setting
        # Parameters
        self.learning_rate = 0.01    # 0.01 this learning rate will be better! Tested
        self.training_epochs = 20
        self.batch_size = 512 
        self.display_step = 1
        # Network Parameters
        self.n_input = 41  
       # hidden layer settings
        self.n_hidden_layers = layers 
        #init layers weights
        nlayers = [self.n_input] + self.n_hidden_layers
        self.ae_params = {
                'encoder':[
                    {
                        'h':tf.Variable(tf.truncated_normal([nlayers[x-1],nlayers[x]],)),
                        'b':tf.Variable(tf.random_normal([nlayers[x]]))
                    } 
                    for x in range(1,len(nlayers))
                ],
                'decoder':[
                    {
                        'h':tf.Variable(tf.truncated_normal([nlayers[x],nlayers[x-1]],)),
                        'b':tf.Variable(tf.random_normal([nlayers[x-1]]))
                    }
                    for x in range(1,len(nlayers))[::-1]
                ]
            }
        # tf Graph input (only pictures)
        self.X = tf.placeholder("float", [None, self.n_input])
        self.encoder_op = MyAutoEncoder.gen_layers(self.X,self.ae_params['encoder'])
        self.decoder_op = MyAutoEncoder.gen_layers(self.encoder_op,self.ae_params['decoder'])
        # Prediction
        y_pred = self.decoder_op
        # Targets (Labels) are the input data.
        y_true = self.X
        # Define loss and optimizer, minimize the squared error
        self.cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        # tf.initialize_all_variables() no long valid from
        # 2017-03-02 if using tensorflow >= 0.12
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        self.tf_sess.run(init)

    @staticmethod
    def gen_layers(x,p):
        layers = [x] + [None]*len(p)
        for i in range(len(p)):
            layers[i+1] = tf.nn.sigmoid(tf.add(tf.matmul(layers[i],p[i]['h']),p[i]['b']))
        return layers[-1]

    def fit(self,data):
        total_batch = int(data.num_examples/self.batch_size)
        # Training cycle
        for epoch in range(self.training_epochs):
            # Loop over all batches
            it = data.get_epoch()
            for i in range(total_batch):
                batch_xs, batch_ys = it.next_batch(self.batch_size)  # max(x) = 1, min(x) = 0
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = self.tf_sess.run([self.optimizer, self.cost], feed_dict={self.X: batch_xs})
            # Display logs per epoch step
            if epoch % self.display_step == 0:
                print("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(c))
        print("Optimization Finished!")

    def transform(self,data):
        return self.tf_sess.run(self.encoder_op,feed_dict={self.X:data})


if __name__ == "__main__":
    print(datetime.datetime.now(),'start...')
    kdd99 = input_data.read_data_sets()
    print(datetime.datetime.now(),'data loaded.')
    svm_train_xs,svm_train_ys = kdd99.train.random_select(10000)
    #tXs,tYs = kdd99.test.random_select(500)

    xs,ys = kdd99.train.random_select(500)
    model = MyAutoEncoder([64,32,16])
    model.fit(kdd99.train)
    print(datetime.datetime.now(),'autoencoder train finished.')
    #encoded_train_xs = model.transform(svm_train_xs)
    encoded_test_xs = model.transform(xs)

    '''
    clf = svm.SVC()
    clf.fit(encoded_train_xs,svm_train_ys)
    print(datetime.datetime.now(),'svm train finished.')
    result = clf.predict(encoded_test_xs)
    print(np.mean(np.equal(result,tYs)))
    '''
    #colors = [list('rbgym')[int(x-1)%5] for x in ys]
    colors = ys
    plt.subplot(2,1,1)
    plt.title('AutoEncoder')
    points = TSNE(random_state=0).fit_transform(encoded_test_xs)
    plt.scatter(points[:,0],points[:,1],c=colors,s=75,alpha=.5)

    points = PCA(n_components=16).fit_transform(xs)
    plt.subplot(2,1,2)
    plt.title('PCA')
    points = TSNE(random_state=0).fit_transform(points)
    plt.scatter(points[:,0],points[:,1],c=colors)

    plt.show()


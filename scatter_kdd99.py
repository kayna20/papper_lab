import matplotlib.pyplot as plt
from datasets.kdd99 import input_data
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import myae
import tensorflow as tf

kdd99 = input_data.read_data_sets()

xs,ys = kdd99.train.random_select(500)

colors = ys

points = TSNE(random_state=0).fit_transform(xs)
plt.subplot(1,3,1)
plt.title('TSNE')
plt.scatter(points[:,0],points[:,1],c=colors,marker='x')

points = PCA(n_components=2).fit_transform(xs)
plt.subplot(1,3,2)
plt.title('PCA')
plt.scatter(points[:,0],points[:,1],c=colors,marker='x')

with tf.Session() as sess:
    model = myae.MyAutoEncoder(sess,[64,32,16,2])
    model.fit(kdd99.train)
    points = model.transform(xs)
    plt.subplot(1,3,3)
    plt.title('AutoEncoder')
    plt.scatter(points[:,0],points[:,1],c=colors,marker='x')

plt.show()


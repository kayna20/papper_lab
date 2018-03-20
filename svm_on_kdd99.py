from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

import numpy as np
import datetime
from datasets.kdd99 import input_data

from myae import MyAutoEncoder

print datetime.datetime.now(),': start..'

kdd99 = input_data.read_data_sets()

print datetime.datetime.now(),': data loaded.train examples num:%s,test examples num:%s' %(kdd99.train.num_examples,kdd99.test.num_examples)
 
Xs,Ys = kdd99.train.random_select()
tXs,tYs = kdd99.test.random_select()

ae = MyAutoEncoder([32,16])
ae.fit(kdd99.train)

ae_Xs = ae.transform(Xs)
ae_tXs = ae.transform(tXs)

clf_ae = svm.SVC()
clf_ae.fit(ae_Xs,Ys)
result_ae = clf_ae.predict(ae_tXs)
ae_cm = confusion_matrix(tYs,result_ae)

print datetime.datetime.now(),': svm precision:%s ' % np.mean(np.equal(result_ae,tYs))
print 'confusion matrix:',ae_cm

pca = PCA(n_components=16)
pca.fit(Xs)

pca_Xs = pca.transform(Xs)
pca_tXs = pca.transform(tXs)

clf_pca = svm.SVC()
clf_pca.fit(pca_Xs,Ys)
result_pca = clf_pca.predict(pca_tXs)
pca_cm = confusion_matrix(tYs,result_pca)

print datetime.datetime.now(),': pca precision:%s ' % np.mean(np.equal(result_pca,tYs))
print 'confusion matrix:',pca_cm


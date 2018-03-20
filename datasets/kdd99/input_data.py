import numpy as np
import collections
import gzip
from os import path
from sklearn import preprocessing

def read_from_text(f):
    data = []
    for line in f:
        data.append(line.strip().split(','))
    return np.array(data)

def transform_label(labels):
    for i in range(len(labels)):
        if labels[i] == 'normal.':
            labels[i] = 0
        else:
            labels[i] = 1

class Prop2NumTranser(object):
    def __init__(self,properties=[]):
        self._properties = properties
    
    def fit(self,data):
        self._properties = list(set(data))

    def transform(self,l):
        for i in range(len(l)):
            l[i] = self._properties.index(l[i])+1

class DataSet(object):
    
    def __init__(self,features,targets):
        assert features.shape[0] == targets.shape[0],(
                'features.shape: %s targets.shape: %s' % (features.shape,targets.shape))
        self._num_examples = features.shape[0]
        self._features = features
        self._targets = targets
        self._index_in_epoch = 0

    @property
    def features(self):
        return self._features
    @property
    def targets(self):
        return self._targets
    @property
    def num_examples(self):
        return self._num_examples

    def random_select(self,n=None):
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        return self._features[perm[:n]],self._targets[perm[:n]]
    
    class Epoch(object):
        def __init__(self,dataset):
            self.dataset = dataset
            self.perm0 = np.arange(dataset.num_examples)
            np.random.shuffle(self.perm0)
            self._index = 0
            self._epoch_complete = False

        def next_batch(self,batch_size):
            if self._epoch_complete:
                return None,None
            start = self._index
            self._index = start + batch_size
            if self._index > self.dataset.num_examples:
                self._epoch_complete = True
            end = self._index
            perm = self.perm0[start:end]
            return self.dataset.features[perm],self.dataset.targets[perm]


    def get_epoch(self):
        return DataSet.Epoch(self)


DataSets = collections.namedtuple('DataSets',['train','test'])

def read_data_sets(data_dir='./data/'):
    TRAIN_FILE = 'kddcup.data_10_percent.gz'
    TEST_FILE = 'corrected.gz'
    module_path = path.dirname(__file__)

    data_path = path.join(module_path,data_dir,TRAIN_FILE)
    with gzip.open(data_path) as f:
        train_data = read_from_text(f)
    
    data_path = path.join(module_path,data_dir,TEST_FILE)
    with gzip.open(data_path) as f:
        test_data = read_from_text(f)
    #transform text properties to number
    transer = Prop2NumTranser()
    #protocal
    transer.fit(np.hstack((train_data[:,1],test_data[:,1])))
    transer.transform(train_data[:,1])
    transer.transform(test_data[:,1])
    #service
    transer.fit(np.hstack((train_data[:,2],test_data[:,2])))
    transer.transform(train_data[:,2])
    transer.transform(test_data[:,2])
    #flag
    transer.fit(np.hstack((train_data[:,3],test_data[:,3])))
    transer.transform(train_data[:,3])
    transer.transform(test_data[:,3])

    #trainsform label
    #transer.fit(np.hstack((train_data[:,-1],test_data[:,-1])))
    #transer.transform(train_data[:,-1])
    #transer.transform(test_data[:,-1])

    transform_label(train_data[:,-1])
    transform_label(test_data[:,-1])

    #to float
    train_data = train_data.astype('float')
    test_data = test_data.astype('float')
    
    #scale to range[0,1]
    min_max_scaler = preprocessing.MinMaxScaler()
    train_data_norm = min_max_scaler.fit_transform(train_data)
    test_data_norm = min_max_scaler.transform(test_data)

    train = DataSet(train_data_norm[:,:-1],train_data[:,-1])
    test = DataSet(test_data_norm[:,:-1],test_data[:,-1])

    return DataSets(train=train,test=test)

if __name__ == '__main__':
    data = read_data_sets()
    print data.train.random_select(100)

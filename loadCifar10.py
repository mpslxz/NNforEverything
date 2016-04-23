import cPickle
import numpy as np
import config

def load():
    f = open("cifar-10-batches-py/data_batch_%d" %config.cifarBatchNumb,'rb')
    d = cPickle.load(f)
    f.close()
    data = d['data']
    return np.reshape(data[0:config.nbOfSamplesFromBatch, 0:(config.imageSize * config.imageSize)],((config.nbOfSamplesFromBatch, config.imageSize, config.imageSize)))

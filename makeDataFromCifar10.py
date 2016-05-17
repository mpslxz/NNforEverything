import loadCifar10
import convolutionKernels
import config
import numpy as np
from scipy.ndimage.filters import convolve

def makeData():
    samples = np.zeros((config.nbOfKernels*config.nbOfSamplesFromBatch, config.imageSize*config.imageSize + config.kernelSize*config.kernelSize))
    targets = np.zeros((config.nbOfKernels*config.nbOfSamplesFromBatch, config.imageSize*config.imageSize))
    kernels = convolutionKernels.makeRandomKernels(nbOfKernels=config.nbOfKernels, kernelSize=config.kernelSize)
    data = loadCifar10.load()
    count = 0
    for i in range(0, config.nbOfSamplesFromBatch):
        for j in range(0, config.nbOfKernels):
            kernel = kernels[j, :, :]
            image = data[i, :, :]/(255*1.0)
            res = convolve(image, kernel, mode='constant', cval=0.0)
            samples[count, :] = np.hstack((np.reshape(image,(1, config.imageSize*config.imageSize)), np.reshape(kernel,(1,config.kernelSize*config.kernelSize))))
            targets[count, :] = np.reshape(res, (1, res.shape[0]*res.shape[1]))

    return samples, targets

if __name__ == "__main__":
    S, T = makeData()
    fS = open("samples",'w')
    fT = open("targets",'w')
    np.save(fS,S)
    np.save(fT,T)
    fS.close()
    fT.close()

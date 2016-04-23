import numpy as np

def makeRandomKernels(nbOfKernels, kernelSize):
    kernels = np.zeros((nbOfKernels, kernelSize, kernelSize))
    for i in range(0, nbOfKernels):
        kernels[i, :, :] = 20 * np.random.random_sample((kernelSize, kernelSize)) - 10
    kernels[0:nbOfKernels/2, :, :] = kernels[0:nbOfKernels/2, :, :]/np.amax(kernels[0:nbOfKernels/2, :, :])
    return kernels

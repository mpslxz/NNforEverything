import models
import config
import numpy as np

if __name__ == "__main__":
    trainingSamples = np.load('samples')
    trainingTargets = np.load('targets')
    m = models.generateMLP(config.numberOfNeuronsPerLayer,config.mlpInputSizeOnCifar10, config.mlpOutputSizeOnCifar10)
    m = models.trainMLP(m,trainingSamples,trainingTargets,epochs=config.nbOfEpochs,batchSize=config.batchSize)
    m.save_weights('convPredictorModel.keras')
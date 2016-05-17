import models
import config
import numpy as np

if __name__ == "__main__":
    trainingSamples = np.load('samples')
    trainingTargets = np.load('targets')
    print np.amax(trainingTargets)
    m = models.generateMLP(config.numberOfNeuronsPerLayer,config.mlpInputSizeOnCifar10, config.mlpOutputSizeOnCifar10)
    m = models.trainMLP(m,trainingSamples,trainingTargets,epochs=config.nbOfEpochs,batchSize=config.batchSize)
    m.save_weights('convPredictorModel.keras')
    # m.load_weights('convPredictorModel.keras')
    # X = trainingSamples[1:1000,:]
    # P = m.predict(X)
    #
    # X = P - trainingTargets[1:1000,:]
    #
    # print np.mean(np.abs(X))
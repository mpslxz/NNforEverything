import config
from keras.models import Sequential
from keras.layers.core import Dense, Activation

def generateMLP(numberOfNeuronsPerLayer, inputSize, outputSize):
    model = Sequential()
    model.add(Dense(numberOfNeuronsPerLayer[0], input_dim=inputSize))
    model.add(Activation('tanh'))
    for i in range(1,len(numberOfNeuronsPerLayer)):
        model.add(Dense(numberOfNeuronsPerLayer[i]))
        model.add(Activation('tanh'))
    model.add(Dense(outputSize))
    # model.add(Activation('sigmoid'))
    model.compile(loss='mean_absolute_error',optimizer='rmsprop')
    return model

def trainMLP(model, trainingSamples, trainingTargets, epochs, batchSize):
    model.fit(trainingSamples,trainingTargets, epochs, batchSize, show_accuracy=True, shuffle=True)
    return model


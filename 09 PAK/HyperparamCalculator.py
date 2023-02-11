from itertools import product
import numpy as np
import random

class HyperparamCalculation():

    def __init__(self,
            iterationsCnt,
            trainData,
            validData,
            setParamFunc,
            seed=100):
        self.bestAccuracy = 0
        self.bestParams = None
        self.iterationsCnt = iterationsCnt
        self.tX, self.tY = trainData
        self.vX, self.vY = validData
        self.parametrize = setParamFunc
        self.seed = seed
    
    @staticmethod
    def calculateAccModel(model, tX, tY):
        predict = np.array(model.predict(tX))
        precision = len(np.where(predict == tY)[0])/len(tY)
        return precision
    
    def printBest(self):
        print(f"Лучшая точность (Validation): {round(self.bestAccuracy*100,2)}%")
        print(f"Лучшие параметры (Validation): {self.bestParams}")
        return self.bestParams

    def getParams(self):
        return self.bestParams

    def nextParamRandom(self, params):
        randomSeeded = random.Random(self.seed)
        for _ in range(self.iterationsCnt):
            yield randomSeeded.choice(params)
    
    def nextParamIter(self, params):
        for param in params:
            yield param

    @staticmethod
    def calculateAverageAccModel(model, trainData, validData, cntIter):
        accs = []
        for _ in range(cntIter):
            model.fit(trainData[0], trainData[1])
            accs.append(HyperparamCalculation.calculateAccModel(model, validData[0], validData[1]))
        return np.mean(accs)

    def calculate(self, params, testIterationsCnt = 15, paramSelector='random'):
        '''
        Подбирает гипер-параметры из словаря params. Стараясь максимизировать точность

        testIterationsCnt - Сколько итераций проводится для расчета средней точности по одному параметру

        paramSelector = "random" - параметры берутся случайным образом (Берутся первые iterationsCnt)
        
        paramSelector = "iter" - Параметры берутся поочереди
        '''
        params = list(product(*params.values()))
        nextParam = self.nextParamRandom
        if paramSelector != 'random':
            nextParam = self.nextParamIter
        for param in nextParam(params):
            model = self.parametrize(param)
            acc = self.calculateAverageAccModel(model, (self.tX, self.tY), (self.vX, self.vY), testIterationsCnt)
            if acc > self.bestAccuracy:
                self.bestAccuracy = acc
                self.bestParams = param
        return self.printBest()


def predictByModel(model, tX, tY, modelName : str = 'None', dataType : str = 'None'):
    precision = HyperparamCalculation.calculateAccModel(model, tX, tY)
    print(f"Точность определения ({dataType}): {round(precision*100,2)}% ({modelName})")
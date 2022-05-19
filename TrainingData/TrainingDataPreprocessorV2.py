import time

import numpy as np
from FeedForward.FeedForwardNeuralNet import Projection_Network
from FeedForward.ProjectionLSTM import Projection_LSTM
from Utils.Util import zeroAppend
from tqdm import tqdm
import progressbar
import torch

class TrainingDataPreprocessorV2:
    __TrainingData = None
    __Maximum_Syntactic_Feature_Size = 0
    __Maximum_TaskElement_Size = 0
    __Maximum_SemanticFeature_Size = 0
    __projected_syntactic_Feature_Size = 2048
    __projected_semantic_Feature_Size = 4096
    __sizeOfVector = __projected_syntactic_Feature_Size + __projected_semantic_Feature_Size

    def __init__(self, maximumSyntacticFeatureSize, maximumTaskElementSize, maximumSemanticFeatureSize,
                 astNode2Vec_size, code2vec_size):
        self.__Maximum_Syntactic_Feature_Size = maximumSyntacticFeatureSize
        self.__Maximum_TaskElement_Size = maximumTaskElementSize
        self.__Maximum_SemanticFeature_Size = maximumSemanticFeatureSize
        self.astNode2Vec_size = astNode2Vec_size
        self.code2vec_size = code2vec_size

    # This method preprocess 'trainingData' such as zero padding syntactic feature vector
    # and generating TrainingData as Dictionary type.
    def preprocess(self, commitList):
        projection_network = Projection_Network(self.__Maximum_Syntactic_Feature_Size,
                                                self.__projected_syntactic_Feature_Size)
        projection_LSTM_network = Projection_LSTM(embedding_size=self.code2vec_size,
                                                  lstm_hidden_size=self.__projected_semantic_Feature_Size)
        summaryList = []
        decoderInputList = []
        decoderOutputList = []
        taskFeatureVectorList = list()

        bar = progressbar.ProgressBar()
        for commit in tqdm(commitList):

            summary = commit['CommitMessage']
            Tasks = commit['TaskData']

            for task in Tasks:
                semanticFeatureList = task['SemanticFeature']
                syntacticDataList = task['SyntacticFeature']
                taskFeatureVector = []
                for taskElement in range(len(semanticFeatureList)):
                    semanticFeature = semanticFeatureList[taskElement]
                    syntacticFeature = np.array(syntacticDataList[taskElement])
                    # zero padding
                    if (len(semanticFeature) <= self.__Maximum_SemanticFeature_Size):
                        semanticFeatureSet = self.appendingZeroToSemanticFeature(semanticFeature)
                        semanticFeatureTensor = torch.tensor(semanticFeatureSet)
                        semanticFeatureTensor = semanticFeatureTensor.unsqueeze(0)
                        semanticFeature = projection_LSTM_network.forward(semanticFeatureTensor)
                        semanticFeature = torch.reshape(semanticFeature, (-1,)).detach().numpy()
                    if (len(syntacticFeature) <= self.__Maximum_Syntactic_Feature_Size):
                        syntacticFeature = zeroAppend(self.__Maximum_Syntactic_Feature_Size, syntacticFeature)
                        syntacticFeature = projection_network.forward(syntacticFeature)
                    taskElementFeatureVector = np.concatenate((semanticFeature, syntacticFeature), axis=None)
                    taskFeatureVector.append(taskElementFeatureVector)

                if (len(semanticFeatureList) < self.__Maximum_TaskElement_Size):
                    gap = self.__Maximum_TaskElement_Size - len(taskFeatureVector)
                    for taskElement in range(gap):
                        featureVector = []
                        featureVector = zeroAppend(self.__sizeOfVector, featureVector)
                        taskFeatureVector.append(featureVector)
                taskFeatureVectorList.append(taskFeatureVector)
                summaryList.append(summary)
                decoderInputList.append('sostoken ' + summary)
                decoderOutputList.append(summary + ' eostoken')
        return taskFeatureVectorList, summaryList, decoderInputList, decoderOutputList

    def appendingZeroToSemanticFeature(self, semanticFeature):
        semanticFeatureSet = []
        for feature in semanticFeature:
            semanticFeatureSet.append(feature)
        size = self.__Maximum_SemanticFeature_Size - len(semanticFeatureSet)
        for i in range(0, size):
            listofzeros = np.zeros(self.code2vec_size, dtype=np.float32)
            semanticFeatureSet.append(listofzeros)
        semanticFeatureSet = np.array(semanticFeatureSet)
        return semanticFeatureSet

    def getTaskElement_VectorSize(self):
        return self.__sizeOfVector

    def getMaximum_TaskElement_Size(self):
        return self.__Maximum_TaskElement_Size

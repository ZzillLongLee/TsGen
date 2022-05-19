from JsonParse.JsonParser import JsonParser
from Utils.Util import zeroAppend


class TrainingDataPreprocessor:
    __TrainingData = None
    __sizeOfVector = 0

    def __init__(self, trainingDataPath):
        parser = JsonParser()
        self.__TrainingData = parser.openJson(trainingDataPath)

    # This method preprocess 'trainingData' such as zero padding syntactic feature vector
    # and generating TrainingDataGenerator as Dictionary type.
    def preprocess(self):
        trainingData = dict()
        MaximumSyntacticFeatureSize = self.__TrainingData['Longest_Syntactic_Feature_Size']
        MaximumTaskElementSize = self.__TrainingData['Longest_Task_Size']
        commitList = self.__TrainingData['CommitData']

        summaryList = list()
        decoderInputList = list()
        decoderOutputList = list()
        taskFeatureVectorList = list()

        for commit in commitList:
            summary = commit['CommitMessage']
            Tasks = commit['TaskData']

            for task in Tasks:
                semanticFeatureList = task['SemanticFeature']
                syntacticDataList = task['SyntacticFeature']
                taskFeatureVector = list()
                for idx in range(len(semanticFeatureList)):
                    semanticFeature = semanticFeatureList[idx]
                    syntacticFeature = syntacticDataList[idx]
                    if (len(syntacticFeature) < MaximumSyntacticFeatureSize):
                        syntacticFeature = zeroAppend(MaximumSyntacticFeatureSize, syntacticFeature)
                    taskElementFeatureVector = semanticFeature + syntacticFeature
                    taskFeatureVector.append(taskElementFeatureVector)
                    # assign the value of size of vector for a task element
                    if (self.__sizeOfVector == 0):
                        self.__sizeOfVector = len(taskElementFeatureVector)

                if (len(taskFeatureVector) < MaximumTaskElementSize):
                    gap = MaximumTaskElementSize - len(taskFeatureVector)
                    for idx in range(gap):
                        featureVector = list()
                        featureVector = zeroAppend(self.__sizeOfVector, featureVector)
                        taskFeatureVector.append(featureVector)

                taskFeatureVectorList.append(taskFeatureVector)
                summaryList.append(summary)
                decoderInputList.append('sostoken ' + summary)
                decoderOutputList.append(summary + ' eostoken')
        trainingData['Encoder FeatureVector'] = taskFeatureVectorList
        trainingData['Summary'] = summaryList
        trainingData['decoder_input'] = decoderInputList
        trainingData['decoder_target'] = decoderOutputList

        return trainingData

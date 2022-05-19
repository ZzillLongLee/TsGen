import pickle

from treelstm import TreeLSTM
from torch import torch
from JsonParse.JsonParser import JsonParser
from treeLstm import TreeLstmDataGenerator
from Word2Vec.Word2VecGenerator import Word2VecGenerator
from Word2Vec.TextPreprocessor import TextPreprocessor
from Utils.Util import poolingSyntacticData
import numpy as np
import json as Json
import pandas as pd
from pathlib import Path
import glob


# from tensorflow.keras.layers.experimental import preprocessing

# This Module is implemented because of the memory issue that handles massive data.
class TrainingDataGeneratorV2:
    astNode2Vec_size = 20
    number_of_vector_code2vec = 30
    __largest_sc_n_words = 0
    __threshhold_task_size = 60

    code2Vec = None
    astNode2Vec = None
    # code2Vec = None

    def __init__(self, trainingDataPath, JsonParser, maximumNumberOfWords, astNode2Vec_size, number_of_vector_code2vec):
        self.parser = JsonParser
        self.trainingDataPath = trainingDataPath
        self.number_of_overSize = 0
        self.__largest_sc_n_words = maximumNumberOfWords
        self.astNode2Vec_size = astNode2Vec_size
        self.number_of_vector_code2vec = number_of_vector_code2vec

    def __stringToList(self, string):
        listRes = list(string.split(" "))
        return listRes

    def __tokenizedCodes(self, sourceCode):
        sourceCodeAsList = self.__stringToList(sourceCode)
        sourceCodeAsList = [x for x in sourceCodeAsList if x != '']
        return sourceCodeAsList

    def __generateSyntacticFeature(self, astNode2Vec, taskElement, astNodeDict):
        tree = TreeLstmDataGenerator.buildTree(astNode2Vec, taskElement, astNodeDict)
        # TreeLSTM
        data = TreeLstmDataGenerator.convert_tree_to_tensors(tree)
        # x, y value must be the same

        model = TreeLSTM(self.astNode2Vec_size, 3).train()
        loss_function = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters())
        # for n in range(1000):
        #     optimizer.zero_grad()
        h, c = model(
            data['features'],
            data['node_order'],
            data['adjacency_list'],
            data['edge_order']
        )
        syntacticData = poolingSyntacticData(h, 'AvgPooling')
        return syntacticData


    def __generateSemanticFeatureWithWord2Vec(self, sourceCodeAsList, code2Vec):
        sourceCodeVectorList = list()
        for word in sourceCodeAsList:
            vector = code2Vec.getVectorValue(word)
            sourceCodeVectorList.append(vector)
        return sourceCodeVectorList

    def __generateSemanticFeatureWithTokenzier(self, sourceCodeAsList, scTokenizer):
        scDict = scTokenizer.word_index
        sourceCodeVectorList = list()
        for word in sourceCodeAsList:
            value = scDict[word.lower()]
            sourceCodeVectorList.append(value)
        semanticFeature = np.array(sourceCodeVectorList)
        semanticFeature = semanticFeature.flatten()
        return semanticFeature

    def trainingDataGenerate(self, projectFilePath, trainingDataPath, code2VecModelPath, ast2VecModelPath,
                             astNodeDictPath):
        project_name = Path(projectFilePath).stem
        json = self.parser.openJson(projectFilePath)
        commits = json

        if (self.astNode2Vec == None and self.code2Vec == None):
            self.astNode2Vec, self.code2Vec = self.__loadWord2vecModels(ast2VecModelPath, code2VecModelPath)
        astNodeDict = self.__loadAstNodeDict(astNodeDictPath)
        self.__generateTrainingData(self.astNode2Vec, astNodeDict, self.code2Vec, commits, project_name)

    def __loadAstNodeDict(self, astNodeDictionaryPath):
        file = open(astNodeDictionaryPath, 'rt', encoding='UTF-8')
        astNodeDict = Json.load(file)
        return astNodeDict

    def __loadModels(self, astNode2VecModelPath, codeTokenizerPath):
        # AST2Vec
        astNode2Vec = Word2VecGenerator()
        astNode2Vec.loadModel(astNode2VecModelPath)
        print("AST2Vec is loaded")
        code2Vec = Word2VecGenerator()
        code2Vec.loadModel(codeTokenizerPath)
        print("AST2Vec is loaded")
        return astNode2Vec, code2Vec

    def __loadWord2vecModels(self, ast2VecModelPath, code2VecModelPath):
        # AST2Vec
        astNode2Vec = Word2VecGenerator()
        astNode2Vec.loadModel(ast2VecModelPath)
        print("AST2Vec is loaded")

        # CODE2VEC
        code2Vec = Word2VecGenerator()
        code2Vec.loadModel(code2VecModelPath)
        print("Code2Vec is loaded")

        return astNode2Vec, code2Vec

    def __generateTrainingData(self, astNode2Vec, astNodeDict, code2Vec, commits, projectName):
        split_size = 100
        longest_task_size = 0
        longest_syntactic_size = 0
        commitList = list()
        index = 0
        trainingDataIndex = 0
        commitSize = len(commits)
        for commit in commits:
            if(commitSize > split_size):
                if (index == split_size):
                    index = 0
                    trainingDataIndex += 1
                    self.generate_TrainingData_As_Json(commitList, longest_syntactic_size, longest_task_size, projectName,
                                                       trainingDataIndex)
                    commitList = list()
                    longest_task_size = 0
                    longest_syntactic_size = 0
                else:
                    index += 1
            else:
                index += 1
                if(commitSize == index):
                    self.generate_TrainingData_As_Json(commitList, longest_syntactic_size, longest_task_size, projectName,
                                                       trainingDataIndex)
            tasks = commit['tasks']
            taskDataList = list()
            for task in tasks:
                taskData = dict()
                taskElementTreeSet = task['taskElementTreeSet']
                syntacticDataList = list()
                semanticDataList = list()
                task_size = len(taskElementTreeSet)
                if (self.__threshhold_task_size > task_size):
                    if(longest_task_size < task_size):
                        longest_task_size = task_size
                    for taskElement in taskElementTreeSet:
                        if (taskElement['n_nodes'] != 0):
                            # Syntactic Feature
                            syntacticFeature = self.__generateSyntacticFeature(astNode2Vec, taskElement, astNodeDict)
                            syntacticFeature_size = len(syntacticFeature)
                            if (longest_syntactic_size < syntacticFeature_size):
                                longest_syntactic_size = syntacticFeature_size
                            syntacticDataList.append(syntacticFeature)
                            # Semantic Feature
                            sourceCode = taskElement['sourceCode']
                            sourceCodeAsList = self.__tokenizedCodes(sourceCode)
                            semanticFeature = self.__generateSemanticFeatureWithWord2Vec(sourceCodeAsList, code2Vec)
                            semanticDataList.append(semanticFeature)
                    taskData['SyntacticFeature'] = syntacticDataList
                    taskData['SemanticFeature'] = semanticDataList
                    taskDataList.append(taskData)
                else:
                    self.number_of_overSize += 1
            commitData = dict()
            commitData['TaskData'] = taskDataList
            commitData['CommitMessage'] = TextPreprocessor.preprocess_sentence(commit['commitMsg'], True)
            commitList.append(commitData)

    def generate_TrainingData_As_Json(self, commitList, longest_syntactic_size, longest_task_size, projectName,
                                      trainingDataIndex):
        trainingData = dict()
        trainingData['CommitData'] = commitList
        trainingData['longest_Task_Size'] = longest_task_size
        trainingData['Longest_Syntactic_Feature_Size'] = longest_syntactic_size
        f = open(self.trainingDataPath + projectName + "_" + str(trainingDataIndex) + ".pkl", "wb")
        pickle.dump(trainingData, f)
        print("Training Sample_Data " + projectName + "_" + str(trainingDataIndex) + " is built")

    def mergeTrainingData(self, trainingDataPath):
        trainingData = dict()
        commitData = list()
        Longest_Syntactic_Feature_Size = 0
        Longest_Task_Size = 0

        for file in glob.glob(trainingDataPath):
            with open(file, 'rb') as f:
                subData = pickle.load(f)
            subCommitData = subData['CommitData']
            commitData.extend(subCommitData)
            syn_feature_size = subData['Longest_Syntactic_Feature_Size']
            if (syn_feature_size > Longest_Syntactic_Feature_Size):
                Longest_Syntactic_Feature_Size = syn_feature_size
            task_size = subData['longest_Task_Size']
            if (task_size > Longest_Task_Size):
                Longest_Task_Size = task_size

        trainingData['CommitData'] = commitData
        trainingData['Longest_Syntactic_Feature_Size'] = Longest_Syntactic_Feature_Size
        trainingData['longest_Semantic_Feature_Size'] = self.__largest_sc_n_words
        trainingData['longest_Task_Size'] = Longest_Task_Size
        return trainingData

    def show_over_size_commits(self):
        print(str(self.number_of_overSize))
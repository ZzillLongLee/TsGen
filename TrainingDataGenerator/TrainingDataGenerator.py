from treelstm import TreeLSTM
from torch import torch
from JsonParse.JsonParser import JsonParser
from treeLstm import TreeLstmDataGenerator
from Word2Vec.Word2VecGenerator import Word2VecGenerator
from Word2Vec.TextPreprocessor import TextPreprocessor
from Utils.Util import zeroAppend
import numpy as np
import json as Json
import glob


# from tensorflow.keras.layers.experimental import preprocessing
class TrainingDataGenerator:
    astNode2Vec_size = 20
    __number_of_vector_code2vec = 40
    __largest_n_words = 0

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
        model = TreeLSTM(20, self.astNode2Vec_size).train()
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
        return h

    def __generateSemanticFeature(self, sourceCodeAsList, code2Vec):
        sourceCodeVectorList = list()
        for word in sourceCodeAsList:
            vector = code2Vec.getVectorValue(word)
            sourceCodeVectorList.append(vector)
        semanticFeature = np.array(sourceCodeVectorList)
        semanticFeature = semanticFeature.flatten()
        # zero padding
        max_len = self.__largest_n_words * self.__number_of_vector_code2vec
        semanticFeature = zeroAppend(max_len, semanticFeature)
        return semanticFeature

    def trainingDataGenerate(self, dataFolderPath, trainingDataPath):
        dataset = []
        for file in glob.glob(dataFolderPath):
            dataset.append(file)

        commits = list()

        parser = JsonParser()
        for data in dataset:
            json = parser.openJson(data)
            commitData = json
            commits.extend(commitData)

        astSentences = list()
        sourceCodeSentences = list()
        astNodeDict = list()

        for commit in commits:
            self.__collectWord2VecData(commit, sourceCodeSentences, astSentences, astNodeDict)
        astNode2Vec, code2Vec = self.__word2vecModelGenerate(sourceCodeSentences, astSentences)
        astNodeDictSet = set(astNodeDict)  # convert it as set data type.
        astNodeDict = list(astNodeDictSet)
        trainingData = self.__generateTrainingData(astNode2Vec, astNodeDict, code2Vec, commits)
        jsonString = Json.dumps(trainingData)

        f = open(trainingDataPath, "w")
        f.write(jsonString)
        f.close()
        print("Training Sample_Data is built")

    def __collectWord2VecData(self, commit, sourceCodeSentences, astSentences, astNodeDict):
        tasks = commit['tasks']
        commitAstNodeDic = commit['astNodeDic']
        astNodeDict.extend(commitAstNodeDic)
        for task in tasks:
            taskElementTreeSet = task['taskElementTreeSet']
            for taskElement in taskElementTreeSet:
                astNodeSentence = taskElement['astNodeSentence']
                astNodeSenAsList = self.__stringToList(astNodeSentence)
                astSentences.append(astNodeSenAsList)

                sourceCode = taskElement['sourceCode']
                sourceCodeAsList = self.__tokenizedCodes(sourceCode)
                sourceCodeSentences.append(sourceCodeAsList)
                if (self.__largest_n_words < len(sourceCodeAsList)):
                    self.__largest_n_words = len(sourceCodeAsList)

    def __word2vecModelGenerate(self, sourceCodeSentences, astSentences):
        # CODE2VEC
        code2Vec = Word2VecGenerator()
        code2Vec.generateModel(sourceCodeSentences, vector_size=self.__number_of_vector_code2vec, window=4, min_count=1,
                               Type='CodeType')
        print("Code2Vec is generated")
        # AST2Vec
        astNode2Vec = Word2VecGenerator()
        astNode2Vec.generateModel(astSentences, vector_size=self.astNode2Vec_size, window=2, min_count=1,
                                  Type="AstType")
        print("AST2Vec is generated")
        return astNode2Vec, code2Vec

    def __generateTrainingData(self, astNode2Vec, astNodeDict, code2Vec, commits):
        longest_task_size = 0
        longest_syntactic_size = 0
        trainingData = dict()
        commitList = list()
        for commit in commits:
            tasks = commit['tasks']
            taskDataList = list()
            for task in tasks:
                taskData = dict()
                taskElementTreeSet = task['taskElementTreeSet']
                syntacticDataList = list()
                semanticDataList = list()
                if (len(taskElementTreeSet) > longest_task_size):
                    longest_task_size = len(taskElementTreeSet)
                for taskElement in taskElementTreeSet:
                    if(taskElement['n_nodes'] != 0):
                        # Syntactic Feature
                        syntacticFeature = self.__generateSyntacticFeature(astNode2Vec, taskElement, astNodeDict)
                        syntacticFeature = syntacticFeature.flatten()
                        syntacticFeatureAsList = syntacticFeature.tolist()
                        syntacticFeature_size = len(syntacticFeatureAsList)
                        if (longest_syntactic_size < syntacticFeature_size):
                            longest_syntactic_size = syntacticFeature_size
                        syntacticDataList.append(syntacticFeatureAsList)
                        # Semantic Feature
                        sourceCode = taskElement['sourceCode']
                        sourceCodeAsList = self.__tokenizedCodes(sourceCode)
                        semanticFeature = self.__generateSemanticFeature(sourceCodeAsList, code2Vec)
                        semanticDataList.append(semanticFeature)
                taskData['SyntacticFeature'] = syntacticDataList
                taskData['SemanticFeature'] = semanticDataList
                taskDataList.append(taskData)
            commitData = dict()
            commitData['TaskData'] = taskDataList
            commitData['CommitMessage'] =  TextPreprocessor.preprocess_sentence(commit['commitMsg'], True)
            commitList.append(commitData)
        trainingData['CommitData'] = commitList
        trainingData["Longest_Syntactic_Feature_Size"] = longest_syntactic_size
        trainingData['Longest_Task_Size'] = longest_task_size
        return trainingData

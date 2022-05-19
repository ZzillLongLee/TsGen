from Word2Vec.Word2VecGenerator import Word2VecGenerator
import glob
from JsonParse.JsonParser import JsonParser
import json as Json


class TrainingComponentGenerator:
    __largest_n_words = 0
    __astNode2Vec_size = 0
    __number_of_vector_code2vec = 0

    def __init__(self, astNode2Vec_size, number_of_vector_code2vec):
        self.__astNode2Vec_size = astNode2Vec_size
        self.__number_of_vector_code2vec = number_of_vector_code2vec

    def generateTrainingComponent(self, dataFolderPath):
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
        self.__word2vecModelGenerate(sourceCodeSentences, astSentences)
        astNodeDictSet = set(astNodeDict)  # convert it as set data type.
        astNodeDict = list(astNodeDictSet)
        jsonString = Json.dumps(astNodeDict)
        with open('Outcome/Models/AstNodeDictionary.json', 'w') as f:
            f.write(jsonString)

        print("Training Components are built")

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
        astNode2Vec.generateModel(astSentences, vector_size=self.__astNode2Vec_size, window=2, min_count=1,
                                  Type="AstType")
        print("AST2Vec is generated")
        return astNode2Vec, code2Vec

    def __stringToList(self, string):
        listRes = list(string.split(" "))
        return listRes

    def __tokenizedCodes(self, sourceCode):
        sourceCodeAsList = self.__stringToList(sourceCode)
        sourceCodeAsList = [x for x in sourceCodeAsList if x != '']
        return sourceCodeAsList

    def getMaximumNumberOfWord(self):
        return self.__largest_n_words

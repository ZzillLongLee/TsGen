import json


class JsonParser(object):

    def openJson(self, filePath):
        file = open(filePath, 'rt', encoding='UTF-8')
        jsonString = json.load(file)
        return jsonString

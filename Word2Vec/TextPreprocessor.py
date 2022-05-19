from bs4 import BeautifulSoup
import re
from Utils.Util import contractions
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

class TextPreprocessor():

  def preprocess_sentence(sentence, remove_stopwords=True):

    stop_words = set(stopwords.words('english'))
    sentence = sentence.lower()  # 텍스트 소문자화
    sentence = BeautifulSoup(sentence, "lxml").text  # <br />, <a href = ...> 등의 html 태그 제거
    sentence = re.sub('"', '', sentence)  # 쌍따옴표 " 제거
    sentence = ' '.join([contractions[t] if t in contractions else t for t in sentence.split(" ")])  # 약어 정규화
    sentence = re.sub(r"'s\b", "", sentence)  # 소유격 제거. Ex) roland's -> roland
    sentence = re.sub('[m]{2,}', 'mm', sentence)  # m이 3개 이상이면 2개로 변경. Ex) ummmmmmm yeah -> umm yeah
    sentence = re.sub("[^a-zA-Z]", " ", sentence)  # 영어 외 문자(숫자, 특수문자 등) 공백으로 변환
    sentence = re.sub(r'\([^)]*\)', '', sentence)  # 괄호로 닫힌 문자열  제거 Ex) my husband (and myself) for => my husband for

    # 불용어 제거 (Text)
    if remove_stopwords:
      tokens = ' '.join(word for word in sentence.split() if not word in stop_words if len(word) > 1)
    # 불용어 미제거 (Summary)
    else:
      tokens = ' '.join(word for word in sentence.split() if len(word) > 1)
    return tokens

  if __name__ == '__main__':
    temp_text = 'HELIX-795 TASK: Drop tasks upon Participant reconnect #6357'
    temp_summary = 'Great way to start (or finish) the day!!!'
    print(preprocess_sentence(temp_text))
    print(preprocess_sentence(temp_summary, 0))
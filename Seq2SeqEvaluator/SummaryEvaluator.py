import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk import word_tokenize
from rouge import Rouge


class SummaryEvaluator():
    bleu = 1
    meteor = 2
    rouge = 3

    def __init__(self):
        nltk.download('punkt')
        nltk.download('wordnet')

    def evaluateSummary(self, evalType, refSummary, targetSummary):
        score = None
        tokenizedRefSummary = word_tokenize(refSummary)
        targetSummary = targetSummary.strip()
        tokenizedTargetSummary = word_tokenize(targetSummary)
        if (evalType == self.bleu):
            # https://donghwa-kim.github.io/BLEU.html BLEU 성능 지표 이해
            score = sentence_bleu([tokenizedRefSummary], tokenizedTargetSummary)
        if (evalType == self.meteor):
            #
            score = meteor_score([tokenizedRefSummary], tokenizedTargetSummary)
        if (evalType == self.rouge):
            # https://huffon.github.io/2019/12/07/rouge/ 로그 성능 지표 이해
            rouge = Rouge()
            score = rouge.get_scores(targetSummary, refSummary)
        return score

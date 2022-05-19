from Seq2Seq.SummaryToSequence import SummaryToSequence
from Seq2Seq.TestSeq2SeqWithAttention import TestSeq2SeqWithAttention
from TrainingDataGenerator.TrainingDataGeneratorV2 import TrainingDataGeneratorV2
from TrainingDataGenerator.TrainingComponentGenerator import TrainingComponentGenerator
from TrainingDataGenerator.TrainingDataPreprocessorV2 import TrainingDataPreprocessorV2
from Utils.Util import getSummaryMaximumLength
from Utils.Util import padSequence
import Seq2SeqEvaluator.SummaryEvaluator as summaryEvaluator
import pandas as pd
import warnings
import pickle
from JsonParse.JsonParser import JsonParser
import glob
import os.path

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

from Seq2Seq.Seq2SeqWithAttention import Seq2SeqWithAttention
import numpy as np

bleu = 1
meteor = 2
rouge = 3

maximumNumberOfWords = 0
astNode2Vec_size = 15
code2vec_size = 25

if __name__ == '__main__':
    dataFilesPath = "Sample_Data/*.json"
    trainingDataPath = "Outcome/TrainingDataGenerator/"
    ast2VecModelPath = "Outcome/Models/AstType.model"
    word2VecModelPath = "Outcome/Models/CodeType.model"
    astNodeDictpath = "Outcome/Models/AstNodeDictionary.json"
    decoder_tokenizer_Path = 'Outcome/Tokenizer/Decoder_Tokenizer.pkl'

    print("-------------------------------------------Generating Training Sample_Data Part-----------------------------------")
    # This part is the generating trainingData from project.json files
    hasTrainingFiles = False
    JsonParser = JsonParser()
    if (os.path.isfile(astNodeDictpath) == False):
        tcg = TrainingComponentGenerator(astNode2Vec_size, code2vec_size)
        tcg.generateTrainingComponent(dataFilesPath)
        maximumNumberOfWords = tcg.getMaximumNumberOfWord();
        print('maximum number of codes: ', maximumNumberOfWords)
    tdg = TrainingDataGeneratorV2(trainingDataPath, JsonParser, maximumNumberOfWords, astNode2Vec_size,
                                  code2vec_size)
    if (hasTrainingFiles == True):
        for file in glob.glob(dataFilesPath):
            tdg.trainingDataGenerate(file, trainingDataPath, word2VecModelPath, ast2VecModelPath, astNodeDictpath)
    tdg.show_over_size_commits()

    print("------------------------------------------Preprocessing Part-----------------------------------------------")
    # Merging each generated training data as a one.
    trainingDataPath = trainingDataPath + "*.pkl"
    fileList = []

    trainingData = tdg.mergeTrainingData(trainingDataPath)

    MaximumSyntacticFeatureSize = trainingData['Longest_Syntactic_Feature_Size']
    MaximumTaskElementSize = trainingData['longest_Task_Size']
    MaximumSemanticFeatureSize = trainingData['longest_Semantic_Feature_Size']
    commitList = trainingData['CommitData']
    splited_data = np.array_split(commitList, 50)
    tdp = TrainingDataPreprocessorV2(MaximumSyntacticFeatureSize, MaximumTaskElementSize, MaximumSemanticFeatureSize,
                                     astNode2Vec_size, code2vec_size)

    encoder_input = list()
    summaries = []
    decoder_input = []
    decoder_target = []
    idx = 0
    for data in splited_data:
        taskFeatureVectorList, summaryList, decoderInputList, decoderOutputList = tdp.preprocess(data)
        encoder_input.extend(taskFeatureVectorList)
        summaries.extend(summaryList)
        decoder_input.extend(decoderInputList)
        decoder_target.extend(decoderOutputList)
        idx += 1
        print("completed preprocess " + str(idx) + ' out of ' + str(len(splited_data)))

    # print('preprocess is done')
    # encoder_input, decoder_input, decoder_target, summaries = suffleDataset(encoder_input, decoder_input, decoder_target,
    #                                                                         summaries)
    encoder_input = np.array(encoder_input, dtype=np.float32)
    print('Converting ndarray is done')

    print("-------------------------------------------Splitting Sample_Data Part---------------------------------------------")
    taskElement_featureVector_len = tdp.getTaskElement_VectorSize()
    task_Maximum_len = tdp.getMaximum_TaskElement_Size()
    max_summary_len = getSummaryMaximumLength(summaries)

    st = SummaryToSequence()

    sequence_decoder_input, sequence_decoder_target = st.convertAsSequence(decoder_input, decoder_target)
    st.saveDecoder_Tokenizer(decoder_tokenizer_Path)
    decoder_vocab_size = st.getVocabularySize()
    print("Decoder Vocabulary Size", decoder_vocab_size)

    sequence_decoder_input = padSequence(sequence_decoder_input, max_summary_len + 1)
    sequence_decoder_target = padSequence(sequence_decoder_target, max_summary_len + 1)

    decoder_vec_size = len(sequence_decoder_input[0, :])

    test_portion_rate = int(len(encoder_input) * 0.032)

    encoder_input_train = encoder_input[:-test_portion_rate]
    decoder_input_train = sequence_decoder_input[:-test_portion_rate]
    decoder_target_train = sequence_decoder_target[:-test_portion_rate]
    summary_train = summaries[:-test_portion_rate]

    encoder_input_eval = encoder_input[-test_portion_rate:]
    decoder_input_eval = sequence_decoder_input[-test_portion_rate:]
    decoder_target_eval = sequence_decoder_target[-test_portion_rate:]
    summary_eval = summaries[-test_portion_rate:]
    evaluation = dict()
    evaluation['encoder'] = encoder_input_eval
    evaluation['decoder'] = decoder_input_eval
    evaluation['summary'] = summary_eval
    f = open('Outcome/evaluation.pkl', 'wb')
    pickle.dump(evaluation, f)
    f.close()

    validation_portion_rate = int(len(encoder_input_train) * 0.1)

    encoder_input_train = encoder_input_train[:-validation_portion_rate]
    decoder_input_train = decoder_input_train[:-validation_portion_rate]
    decoder_target_train = decoder_target_train[:-validation_portion_rate]

    encoder_input_valid = encoder_input_train[-validation_portion_rate:]
    decoder_input_valid = decoder_input_train[-validation_portion_rate:]
    decoder_target_valid = decoder_target_train[-validation_portion_rate:]

    print("-----------------------------------------------Training Part-----------------------------------------------")
    seq2seq = Seq2SeqWithAttention()
    hidden_size = 256
    seq2seq.buildModel(taskElement_featureVector_len, hidden_size, MaximumTaskElementSize, decoder_vocab_size,
                       decoder_vec_size)
    print("Model is built!")

    print("Training Model....")
    model, history = seq2seq.trainModel(encoder_input_train, decoder_input_train,
                                        decoder_target_train, validation_encoder_input=encoder_input_valid,
                                        validation_decoder_input=decoder_input_valid,
                                        validation_decoder_target=decoder_target_valid)
    print("Training is done")

    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.legend()
    # plt.show()

    testSeq2SeqModel = TestSeq2SeqWithAttention(None, None)
    encoder_inputs, encoder_outputs, state_h, state_c = seq2seq.getEncoderValues()
    testSeq2SeqModel.buildEncoder(encoder_inputs, encoder_outputs, state_h, state_c)
    decoder_inputs, dec_emb_layer, decoder_lstm, attn_layer, decoder_softmax_layer = seq2seq.getDecoderValues()
    testSeq2SeqModel.buildDecoder(decoder_inputs, hidden_size, task_Maximum_len, dec_emb_layer, decoder_lstm,
                                  attn_layer, decoder_softmax_layer)
    testSeq2SeqModel.saveModels('Outcome/Encoder_model.h5', 'Outcome/Decoder_model.h5')
    print("Model is saved")

    print("-------------------------------------------Evaluation Part---------------------------------------------------")
    testSeq2SeqModel = None
    st = None

    if (testSeq2SeqModel == None and st == None):
        testSeq2SeqModel = TestSeq2SeqWithAttention(None, None)
        st = SummaryToSequence()

    evaluationData_path = 'Outcome/evaluation.pkl'
    with open(evaluationData_path, 'rb') as handle:
        evaluationSet = pickle.load(handle)
    encoder_input_eval = evaluationSet['encoder']
    summary_eval = evaluationSet['summary']

    testSeq2SeqModel.loadModel('Outcome/Encoder_model.h5', 'Outcome/Decoder_model.h5')
    st.loadDecoder_Tokenizer(decoder_tokenizer_Path)
    tokenizer = st.getTokenizer()
    tar_word_to_index = tokenizer.word_index  # 요약 단어 집합에서 단어 -> 정수를 얻음
    tar_index_to_word = tokenizer.index_word  # 요약 단어 집합에서 정수 -> 단어를 얻음
    print("Test is ready")
    dictionary_List = list()
    summaryEvaluator = summaryEvaluator.SummaryEvaluator()
    for i in range(0, len(encoder_input_eval)):
        generatedSummary = testSeq2SeqModel.decode_sequence(
            encoder_input_eval[i].reshape(1, task_Maximum_len, taskElement_featureVector_len), tar_word_to_index,
            tar_index_to_word,
            max_summary_len)

        evaluationDic = dict()
        realSummary = summary_eval[i]
        if (realSummary != ''):
            evaluationDic['reference summary'] = realSummary
            evaluationDic['candidate summary'] = generatedSummary
            bleu_score = summaryEvaluator.evaluateSummary(bleu, generatedSummary, realSummary)
            evaluationDic['bleu score'] = bleu_score
            meteor_score = summaryEvaluator.evaluateSummary(meteor, generatedSummary, realSummary)
            evaluationDic['meteor score'] = meteor_score
            rouge_score = summaryEvaluator.evaluateSummary(rouge, generatedSummary, realSummary)
            rouge_1 = rouge_score[0]['rouge-1']
            evaluationDic['rouge-1 precision'] = rouge_1['p']
            evaluationDic['rouge-1 recall'] = rouge_1['r']
            evaluationDic['rouge-1 f-score'] = rouge_1['f']
            rouge_2 = rouge_score[0]['rouge-2']
            evaluationDic['rouge-2 precision'] = rouge_2['p']
            evaluationDic['rouge-2 recall'] = rouge_2['r']
            evaluationDic['rouge-2 f-score'] = rouge_2['f']
            rouge_l = rouge_score[0]['rouge-l']
            evaluationDic['rouge-l precision'] = rouge_l['p']
            evaluationDic['rouge-l recall'] = rouge_l['r']
            evaluationDic['rouge-l f-score'] = rouge_l['f']
            if (bleu_score != 0):
                dictionary_List.append(evaluationDic)
            print('generated ' + str(i) + ' out of the ' + str(len(encoder_input_eval)))

    df = pd.DataFrame(dictionary_List)
    df.to_csv('Outcome/evaluation.csv')
    print("Test Outcome is generated")

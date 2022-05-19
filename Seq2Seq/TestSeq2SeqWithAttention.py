import tensorflow.keras.models
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
import numpy as np
import json

from Seq2Seq.Attention import AttentionLayer


class TestSeq2SeqWithAttention():

    def __init__(self, encoder_model, decoder_model):
        if(encoder_model == None and decoder_model == None):
            self.__encoder_model = None
            self.__decoder_model = None
        else:
            self.__encoder_model = encoder_model
            self.__decoder_model = decoder_model

    def buildEncoder(self, encoder_inputs, encoder_outputs, state_h, state_c):
        if (self.__encoder_model == None):
            # 인코더 설계
            self.__encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])
        else:
            print("The encoder model is already existed! Please use decodeSequence function!")

    def buildDecoder(self, decoder_inputs, hidden_size, task_max_len, dec_emb_layer, decoder_lstm, attn_layer, decoder_softmax_layer):
        if (self.__decoder_model == None):
            # 이전 시점의 상태들을 저장하는 텐서
            decoder_state_input_h = Input(shape=(hidden_size,))
            decoder_state_input_c = Input(shape=(hidden_size,))

            dec_emb2 = dec_emb_layer(decoder_inputs)
            # 문장의 다음 단어를 예측하기 위해서 초기 상태(initial_state)를 이전 시점의 상태로 사용. 이는 뒤의 함수 decode_sequence()에 구현
            # 훈련 과정에서와 달리 LSTM의 리턴하는 은닉 상태와 셀 상태인 state_h와 state_c를 버리지 않음.
            decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h,
                                                                                         decoder_state_input_c])

            # 어텐션 함수
            decoder_hidden_state_input = Input(shape=(task_max_len, hidden_size))
            attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
            decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

            # 디코더의 출력층
            decoder_outputs2 = decoder_softmax_layer(decoder_inf_concat)

            # 최종 디코더 모델
            self.__decoder_model = Model(
                [decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
                [decoder_outputs2] + [state_h2, state_c2])
        else:
            print("The decoder model is already existed! Please use decodeSequence function!")

    def saveModels(self, encoderPath, decoderPath):
        if(self.__encoder_model != None and self.__decoder_model != None):
            self.__encoder_model.save(encoderPath)
            self.__decoder_model.save(decoderPath)

    def loadModel(self, encoderModelPath,  decoderModelPath):
        self.__encoder_model = tensorflow.keras.models.load_model(encoderModelPath)
        self.__decoder_model = tensorflow.keras.models.load_model(decoderModelPath, custom_objects={"AttentionLayer": AttentionLayer})

    def decode_sequence(self, input_value, tar_word_to_index, tar_index_to_word, summary_max_len):
        # 입력으로부터 인코더의 상태를 얻음
        e_out, e_h, e_c = self.__encoder_model.predict(input_value)

        # <SOS>에 해당하는 토큰 생성
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = tar_word_to_index['sostoken']

        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:  # stop_condition이 True가 될 때까지 루프 반복

            output_tokens, h, c = self.__decoder_model.predict([target_seq] + [e_out, e_h, e_c])
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = tar_index_to_word[sampled_token_index]

            if (sampled_token != 'eostoken'):
                decoded_sentence += ' ' + sampled_token

            #  <eos>에 도달하거나 최대 길이를 넘으면 중단.
            if (sampled_token == 'eostoken' or len(decoded_sentence.split()) >= (summary_max_len - 1)):
                stop_condition = True

            # 길이가 1인 타겟 시퀀스를 업데이트
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # 상태를 업데이트 합니다.
            e_h, e_c = h, c

        return decoded_sentence
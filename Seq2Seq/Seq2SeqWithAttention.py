from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from Seq2Seq.Attention import AttentionLayer


class Seq2SeqWithAttention():

    def __init__(self):
        self.__model = None
        self.__encoder_inputs = None
        self.__encoder_outputs = None
        self.__encoder_state_h = None
        self.__encoder_state_c = None
        self.__decoder_inputs = None
        self.__dec_emb_layer = None
        self.__decoder_lstm = None
        self.__attn_layer = None
        self.__decoder_softmax_layer = None

    def buildModel(self, featureVector_dim, hidden_size, maximum_taskElement_len, decoder_vocab, decoder_data_dim):
        # 인코더
        self.__encoder_inputs = Input(shape=(maximum_taskElement_len, featureVector_dim))

        # 인코더의 LSTM 1
        encoder_lstm1 = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4,
                             input_shape=(maximum_taskElement_len, featureVector_dim))
        encoder_output1, state_h1, state_c1 = encoder_lstm1(self.__encoder_inputs)

        # 인코더의 LSTM 2
        encoder_lstm2 = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
        encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

        # 인코더의 LSTM 3
        encoder_lstm3 = LSTM(hidden_size, return_state=True, return_sequences=True, dropout=0.4, recurrent_dropout=0.4)
        self.__encoder_outputs, self.__encoder_state_h, self.__encoder_state_c = encoder_lstm3(encoder_output2)

        # 디코더
        self.__decoder_inputs = Input(shape=(None,))

        # 디코더의 임베딩 층
        self.__dec_emb_layer = Embedding(decoder_vocab, decoder_data_dim)
        dec_emb = self.__dec_emb_layer(self.__decoder_inputs)

        # 디코더의 LSTM
        self.__decoder_lstm = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4,
                                   recurrent_dropout=0.2)
        decoder_outputs, _, _ = self.__decoder_lstm(dec_emb,
                                                    initial_state=[self.__encoder_state_h, self.__encoder_state_c])

        # 어텐션 층(어텐션 함수)
        self.__attn_layer = AttentionLayer(name='attention_layer')
        attn_out, attn_states = self.__attn_layer([self.__encoder_outputs, decoder_outputs])

        # 어텐션의 결과와 디코더의 hidden state들을 연결
        decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

        # 디코더의 출력층
        self.__decoder_softmax_layer = Dense(decoder_vocab, activation='softmax')
        decoder_softmax_outputs = self.__decoder_softmax_layer(decoder_concat_input)

        # 모델 정의
        self.__model = Model([self.__encoder_inputs, self.__decoder_inputs], decoder_softmax_outputs)
        self.__model.summary()
        self.__model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

    def trainModel(self, encoder_input_train, decoder_input_train, decoder_target_train, validation_encoder_input=None,
                   validation_decoder_input=None, validation_decoder_target=None):
        if self.__model == None:
            print('The model is None! you should build the model first.')
            return None
        else:
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
            history = self.__model.fit(x=[encoder_input_train, decoder_input_train], y=decoder_target_train, \
                                       validation_data=(
                                       [validation_encoder_input, validation_decoder_input], validation_decoder_target),
                                       batch_size=256, callbacks=[es], epochs=150)
            return self.__model, history

    def getEncoderValues(self):
        return self.__encoder_inputs, self.__encoder_outputs, self.__encoder_state_h, self.__encoder_state_c

    def getDecoderValues(self):
        return self.__decoder_inputs, self.__dec_emb_layer, self.__decoder_lstm, self.__attn_layer, self.__decoder_softmax_layer

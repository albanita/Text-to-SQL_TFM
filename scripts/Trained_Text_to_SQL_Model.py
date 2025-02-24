import numpy as np
import re
import string
import json
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, AdditiveAttention, Concatenate, Dense, LayerNormalization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json


nltk.download('stopwords')
nltk.download('wordnet')


class Trained_Text_to_SQL_Model:

    def __init__(self, model_path: str, input_tokenizer_path: str, target_tokenizer_path: str):
        self.model_path = model_path
        self.input_tokenizer_path = input_tokenizer_path
        self.target_tokenizer_path = target_tokenizer_path

        self.__load_trained_model_and_tokens()
        self.__extract_encoder_decoder_architecture()

    def __load_trained_model_and_tokens(self):
        # Load modelul cu custom objects
        self.__model = load_model(self.model_path, safe_mode=False)

        # Load the input tokenizer
        with open(self.input_tokenizer_path, 'r') as f:
            input_tokenizer_config = json.load(f)
        self.__input_tokenizer = tokenizer_from_json(json.dumps(input_tokenizer_config))

        # Load the target tokenizer
        with open(self.target_tokenizer_path, 'r') as f:
            target_tokenizer_config = json.load(f)
        self.__target_tokenizer = tokenizer_from_json(json.dumps(target_tokenizer_config))

    def __extract_encoder_decoder_architecture(self):
        # Encoder
        encoder_inputs = self.__model.input[0]
        encoder_embedding = self.__model.get_layer("embedding")(encoder_inputs)
        encoder_norm = self.__model.get_layer("layer_normalization")(encoder_embedding)
        encoder_lstm = self.__model.get_layer("lstm")
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_norm)

        self.__encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])
        print("✅ Encoder model successfully extracted!")

        # Decoder
        decoder_inputs = self.__model.input[1]
        decoder_embedding = self.__model.get_layer("embedding_1")(decoder_inputs)
        decoder_norm = self.__model.get_layer("layer_normalization_1")(decoder_embedding)

        decoder_lstm = self.__model.get_layer("lstm_1")
        decoder_state_input_h = Input(shape=(512,), name="decoder_input_h")
        decoder_state_input_c = Input(shape=(512,), name="decoder_input_c")
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_lstm_outputs, state_h, state_c = decoder_lstm(
            decoder_norm, initial_state=decoder_states_inputs
        )

        encoder_outputs_input = Input(shape=(100, 512), name="encoder_outputs_input")
        attention_layer = self.__model.get_layer("additive_attention")
        attention_outputs = attention_layer([decoder_lstm_outputs, encoder_outputs_input])

        decoder_combined_context = Concatenate(axis=-1)([attention_outputs, decoder_lstm_outputs])
        decoder_dense = self.__model.get_layer("dense")
        decoder_outputs = decoder_dense(decoder_combined_context)

        self.__decoder_model = Model(
            [decoder_inputs, encoder_outputs_input] + decoder_states_inputs,
            [decoder_outputs, state_h, state_c]
        )

        print("✅ Decoder model successfully extracted!")


        # encoder_inputs = self.__model.input[0]  # input_7 (Encoder input)
        # embedding_enc = self.__model.get_layer("embedding_6")(encoder_inputs)
        # norm_enc = self.__model.get_layer("layer_normalization_6")(embedding_enc)
        # lstm_1, state_h, state_c = self.__model.get_layer("lstm_11")(norm_enc)
        # encoder_outputs, state_h_enc, state_c_enc = self.__model.get_layer("lstm_12")(lstm_1)
        #
        # # Define the encoder model
        # self.__encoder_model = Model(encoder_inputs, [encoder_outputs, state_h_enc, state_c_enc])
        # print("✅ Encoder model successfully extracted!")
        #
        # # Decoder model
        # decoder_inputs = self.__model.input[1]  # input_8 (Decoder input)
        # embedding_dec = self.__model.get_layer("embedding_7")(decoder_inputs)
        # norm_dec = self.__model.get_layer("layer_normalization_7")(embedding_dec)
        # decoder_state_input_h = Input(shape=(256,), name="decoder_state_input_h")
        # decoder_state_input_c = Input(shape=(256,), name="decoder_state_input_c")
        #
        # decoder_lstm_1 = self.__model.get_layer("lstm_13")
        # decoder_outputs, state_h_dec, state_c_dec = decoder_lstm_1(norm_dec, initial_state=[decoder_state_input_h,
        #                                                                                     decoder_state_input_c])
        #
        # decoder_lstm_2 = self.__model.get_layer("lstm_14")
        # decoder_outputs, state_h_dec, state_c_dec = decoder_lstm_2(decoder_outputs,
        #                                                            initial_state=[state_h_dec, state_c_dec])
        #
        # attention = self.__model.get_layer("additive_attention_2")
        # attention_output = attention([decoder_outputs, encoder_outputs])
        # concat = self.__model.get_layer("concatenate_1")
        # decoder_combined_context = concat([attention_output, decoder_outputs])
        #
        # decoder_dense = self.__model.get_layer("dense_1")
        # decoder_outputs = decoder_dense(decoder_combined_context)
        #
        # # Define the decoder model
        # self.__decoder_model = Model(
        #     [decoder_inputs, encoder_outputs, decoder_state_input_h, decoder_state_input_c],
        #     [decoder_outputs, state_h_dec, state_c_dec]
        # )
        # print("✅ Decoder model successfully extracted!")

    def __decode_sequence(self, input_seq, max_target_length=100):
        encoder_outputs, state_h, state_c = self.__encoder_model.predict(input_seq)

        start_token = self.__target_tokenizer.word_index.get('<start>', 1)
        end_token = self.__target_tokenizer.word_index.get('<end>', 2)

        target_seq = np.array([[start_token]])
        decoded_sentence = []

        for _ in range(max_target_length):
            output_tokens, state_h, state_c = self.__decoder_model.predict(
                [target_seq, encoder_outputs, state_h, state_c]
            )

            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = self.__target_tokenizer.index_word.get(sampled_token_index, '<unk>')

            if sampled_word == '<end>':
                break

            decoded_sentence.append(sampled_word)
            target_seq = np.array([[sampled_token_index]])

        return ' '.join(decoded_sentence)

    def __preprocess_input_text(self, text, max_length=100):
        text = text.lower()

        # Remove punctuation marks
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)

        # Stopwords and lematization
        custom_stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        words = [lemmatizer.lemmatize(word) for word in text.split() if word not in custom_stop_words]
        cleaned_text = ' '.join(words)

        # Convert to sequence and padding
        sequence = self.__input_tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

        return padded_sequence

    def generate_sql(self, input_nl_query: str) -> str:
        input_seq = self.__preprocess_input_text(input_nl_query)
        return self.__decode_sequence(input_seq)

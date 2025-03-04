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
        # ----------------------------- #
        #          ENCODER              #
        # ----------------------------- #
        # Define encoder inputs
        encoder_inputs = self.__model.get_layer("input_7").input
        encoder_embedding = self.__model.get_layer("embedding_6")(encoder_inputs)
        encoder_norm = self.__model.get_layer("layer_normalization_6")(encoder_embedding)

        # LSTM layers
        encoder_lstm1 = self.__model.get_layer("lstm_11")
        encoder_lstm2 = self.__model.get_layer("lstm_12")

        # Decoder inputs
        decoder_inputs = self.__model.get_layer("input_8").input
        decoder_embedding = self.__model.get_layer("embedding_7")(decoder_inputs)
        decoder_norm = self.__model.get_layer("layer_normalization_7")(decoder_embedding)

        # Decoder LSTMs with states from the encoder
        decoder_lstm1 = self.__model.get_layer("lstm_13")
        decoder_lstm2 = self.__model.get_layer("lstm_14")

        attention_layer = self.__model.get_layer("additive_attention_2")
        decoder_dense = self.__model.get_layer("dense_1")

        encoder_outputs1, self.__model, state_c1 = encoder_lstm1(encoder_norm)

        encoder_outputs2, state_h2, state_c2 = encoder_lstm2(encoder_outputs1)  # This is the final output

        # Define encoder model (ensure it returns 3 values)
        self.__encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs2, state_h2, state_c2])

        print("✅ Encoder model successfully extracted!")

        # ----------------------------- #
        #          DECODER              #
        # ----------------------------- #
        # NEW input layers to avoid duplication errors

        # Define decoder initial states (received from the encoder during inference)
        decoder_state_input_h1 = Input(shape=(256,))
        decoder_state_input_c1 = Input(shape=(256,))
        decoder_state_input_h2 = Input(shape=(256,))
        decoder_state_input_c2 = Input(shape=(256,))

        decoder_outputs1, state_h1, state_c1 = decoder_lstm1(
            decoder_norm, initial_state=[decoder_state_input_h1, decoder_state_input_c1]
        )
        decoder_outputs2, state_h2, state_c2 = decoder_lstm2(
            decoder_outputs1, initial_state=[decoder_state_input_h2, decoder_state_input_c2]
        )

        # Attention mechanism (use `lstm_12` output instead of `encoder_lstm1`)

        encoder_outputs = Input(shape=(100, 256))  # Corrected to match decoder_outputs2
        attention_output = attention_layer([decoder_outputs2, encoder_outputs])

        # Concatenation and final dense layer
        concatenated = Concatenate()([attention_output, decoder_outputs2])

        decoder_outputs = decoder_dense(concatenated)

        # Decoder model
        self.__decoder_model = Model(
            [decoder_inputs, decoder_state_input_h1, decoder_state_input_c1, decoder_state_input_h2,
             decoder_state_input_c2, encoder_outputs],
            [decoder_outputs, state_h1, state_c1, state_h2, state_c2]
        )

        print("✅ Decoder model successfully extracted!")

    def __decode_sequence(self, input_seq, max_target_length=100):
        """Generate SQL query from a natural language input."""
        # Get encoder outputs and states
        encoder_outputs, state_h1, state_c1 = self.__encoder_model.predict(input_seq)

        # Start token ID
        start_token = self.__target_tokenizer.word_index['<start>']
        end_token = self.__target_tokenizer.word_index['<end>']

        # Initial decoder input (start token)
        target_seq = np.array([[start_token]])

        # Initialize decoder states (use both LSTM layers' states)
        state_h2, state_c2 = state_h1, state_c1  # Assuming the second layer starts with same states

        decoded_sentence = []

        for _ in range(max_target_length):
            # Predict next token with all 6 required inputs
            decoder_inputs = [target_seq, state_h1, state_c1, state_h2, state_c2, encoder_outputs]
            predictions, state_h1, state_c1, state_h2, state_c2 = self.__decoder_model.predict(decoder_inputs)

            # Get most probable word index
            sampled_token_index = np.argmax(predictions[0, -1, :])
            sampled_word = self.__target_tokenizer.index_word.get(sampled_token_index, '<unk>')

            if sampled_word == '<end>':
                break

            decoded_sentence.append(sampled_word)

            # Update target sequence with the new predicted token
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

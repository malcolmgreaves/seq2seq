# FROM:
# https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/

from typing import *

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.utils.vis_utils import plot_model


def char_seq2seq_model(num_encoder_tokens:int, 
                       num_decoder_tokens:int,
                       latent_dim: int,
                       activation_fn:str = 'softmax') -> Model:
  # input character sequence
  encoder_inputs = Input(shape=(None, num_encoder_tokens))
  
  # encoder model: accepts input, output is read by decoder
  encoder = LSTM(latent_dim, return_state=True)
  _, state_hidden, state_context = encoder(encoder_inputs)
  encoder_states = [state_hidden, state_context]

  # decoder input is encoder hidden, context states
  decoder_inputs = Input(shape=(None, num_decoder_tokens))

  # The decoder is condigured to return full output sequences.
  # The internal decoder states are *not* used during training:
  # they are used during inference (note: this is teacher-forcing learning).
  decoder = LSTM(latent_dim, return_sequences=True, return_state=True)
  decoder_outputs, _, _ = decoder(decoder_inputs, intitial_state=encoder_states)
  decoder_dense = Dense(num_decoder_tokens, activation=activation_fn)
  decoded_char_outputs = decoder_dense(decoder_outputs)

  # Seq2Seq model is defined using the encoder inputs, decoder inputs, and
  # the final decoded character sequence outputs.
  seq2seq_model = Model([encoder_inputs, decoder_inputs], decoded_char_outputs)
  return seq2seq_model


def exp_eng_french_char_seq2seq_model(labeled_translations_fi:str = 'data/fra.txt',
                                      num_encoder_tokens:int = 71,
                                      num_decoder_tokens:int = 93,
                                      latent_dim:int = 256,
                                      model_diagram_fi:str = "diagram-char_seq2seq_model.png"):
  model = char_seq2seq_model(
    num_encoder_tokens=num_encoder_tokens,
    num_decoder_tokens=num_decoder_tokens,
    latent_dim=latent_dim,
    activation_fn='softmax',
  )
  plot_model(model, to_file=model_diagram_fi, show_shapes=True)

  raise NotImplementedError("Need to implement training call & evaluation")


if __name__ == "__main__":
  print("Training character-based sequence-to-sequence translation model on English <--> French "
        "sentences.")
  exp_eng_french_char_seq2seq_model()
  print("Done training.")
  
# seq2seq

Playground for learning about sequence 2 sequence models. Specifically the encoder-decoder 
framework.

## English <> French Translation

Data is located under `data/fra.txt`: english sentence, tab, french translation. The data's
[README](data/README.md) explains more.

The `tutorial_encode_decode.py` module creates an encoder-decoder LSTM network and performs 
equence-to-sequence tranlation. It uses the above data to demonstrate the network's performance.

This model is a _character_-based RNN: it translates character sequences, not word or token
sequences.


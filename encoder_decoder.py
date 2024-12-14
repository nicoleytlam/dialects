import torch
from typing import Optional, Union, Generator, List, Tuple


import numpy as np
import torch
import torch.nn as nn

import random

class RNNEncoder(nn.Module):
    """
    RNN encoder.
    """

    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int, 
                 output_dim: int, 
                 recurrent_type: str = 'rnn',
                 bidirectional: bool = True,                 
                 dropout_prob: float = .5):
        """
        :param vocab_size: The size of the model vocabulary
        :param embedding_size: The size of the word embeddings used by
            the model
        :param hidden_size: The size of the RNN's hidden state vector
        :param output_dim The size of the encoder's output vector
        :param recurrent_type: The type of recurrent unit, either 'gru' or 'rnn'
        :param bidirectional: Whether or not the RNN is bidirectional
        :param dropout_prob: The dropout probability used in the encoder
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)
        
        #Problem 10: Replace the following lines with your own code. Do
        #not edit anything above this line in this function.
        if recurrent_type == 'rnn':
            self.rnn = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, 
                              batch_first=True, bidirectional=bidirectional)
        elif recurrent_type == 'gru':
            self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, 
                              batch_first=True, bidirectional=bidirectional)
        else:
            raise ValueError("recurrent_type must be either 'rnn' or 'gru'")

        rnn_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.linear = nn.Linear(rnn_output_size, output_dim)

        self.dropout = nn.Dropout(dropout_prob)

        
    def forward(self, sentences: torch.LongTensor) -> torch.Tensor:
        """
        :param sentences: A mini-batch of sentences, not including [BOS]
            but padded with [PAD], to be processed by the encoder.
            Shape: (batch size, sentence length)
        :return: outputs: the encoding for each of the tokens in each sentence in 
                          the batch
                    Shape: (batch size, sentence length,        
                            output dim)                          
                 final_hidden: the encoding of the final token of each sentence. 
                    Shape: (1, batch size, output dim)
        """
        # Problem 11: Replace the following lines with your code.
        e = self.dropout(self.embedding_layer(sentences))
        o, h_n = self.rnn(e)
        outputs = self.linear(o)

        if self.bidirectional:
            final_hidden = self.linear(torch.reshape(h_n, (1, -1, self.hidden_size * 2)))
        else:
            final_hidden = self.linear(h_n)

        return outputs, final_hidden
    



class Attention(nn.Module):
    """
    Attention module
    """
    def __init__ (self, encoder_dimension: int, decoder_dimension: int,     
                  attention_dimension: int):
        """
        :param encoder_dimension: the size of the encoder embeddings (used to create key vectors)
        :param decoder_dimension: the size of the decoder embeddings (used to create query vectors)
        :param attention_dimension: the size of the vectors produced by the Q and V mappings whose dot product is taken
        """
        super().__init__()
        #Problem 12: Replace the following lines with your code.
        self.Q = nn.Linear(decoder_dimension, attention_dimension)
        self.K = nn.Linear(encoder_dimension, attention_dimension)
        self.V = nn.Linear(encoder_dimension, decoder_dimension)

    def forward(self, query_embed: torch.Tensor, embed_sequence: torch.Tensor):
        """
        :param query_embed: the embedding that will be used to form the query vector
            Shape: (batch size, decoder dimension)
        :param embed_sequence: the sequence of embeddings used to create the key    
            vectors
            Shape: (batch size, sequence length, encoder dimension)
        :return: weighted_sum: the vectors created by the sum of the value vectors 
                        weighted by the attention weights
                    Shape: (batch size, 1, decoder dimension)
                attention_weights: the attention weights assigned to each position 
                        in the input sequence
                    Shape: (batch size, 1, input length)
        """
        #Problem 13: Replace the following lines with your code 
        q = self.Q(query_embed)
        k = self.K(embed_sequence)
        v = self.V(embed_sequence)

        # Calculate attention scores
        # q (batch size, attention dim)
        # k (batch size, sequence length, attention dim)
        # v (batch size, sequence length, decoder dimension)
        product = torch.bmm(q.unsqueeze(1), k.permute(0, 2, 1)) 
        product = product / np.sqrt(product.shape[2]) # Shape: (batch_size, 1, sequence_length)
        attention_weights = torch.softmax(product, dim=2)  # Shape: (batch_size, 1, sequence_length)

        # Compute weighted sum in the original decoder dimension
        weighted_sum = torch.bmm(attention_weights, v)  # Shape: (batch_size, 1, decoder_dimension)

        return weighted_sum, attention_weights
    
class RNNDecoder(nn.Module):
    """
    RNN Sequence Decoder
    """

    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int,
                 attention: Attention, recurrent_type: str = 'rnn',
                 use_attention: bool = True,
                 dropout_prob: float = .5):
        """
        :param vocab_size: The size of the output vocabulary
        :param embedding_size: The size of the word embeddings used by
            the model
        :param hidden_size: The size of the RNN's hidden state vector
        :param attention: An attention module
        :param recurrent_type: The kind of recurrent unit to be used ('rnn' or 'gru')
        :param: use_attention: If True, then the decoder uses the attention module for its computation
        :param: dropout_prob: The dropout probability
        """
        super().__init__()
        self.attention = attention
        self.use_attention = use_attention
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)

        # Problem 14: Replace the following two lines with your own
        # code. Do not edit anything above this line in this function.
        rnn_input_size = embedding_size + (hidden_size if use_attention else 0)

        if recurrent_type == 'rnn':
            self.rnn = nn.RNN(input_size=rnn_input_size, hidden_size=hidden_size, 
                              batch_first=True)
        elif recurrent_type == 'gru':
            self.rnn = nn.GRU(input_size=rnn_input_size, hidden_size=hidden_size, 
                              batch_first=True)
        else:
            raise ValueError("recurrent_type must be either 'rnn' or 'gru'")
        
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(hidden_size, embedding_size)


    def forward(self, input: torch.LongTensor, previous_hidden: 
                torch.LongTensor, 
                encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        :param input: batch of tokens to be presented to the decoder
            Shape: (batch size)
        :param previous hidden: batch of previous hidden vectors to be given as input to the recurrent unit
            Shape: (1, batch size, decoder hidden size)
        :param encoder_outputs: batch of encoder outputs
            Shape: (batch size, sequence length, encoder hidden size)
        :return logits: the logits for each possible output symbol 
                    Shape: (batch size, vocab size)
                h: the hidden state resulting from the current state of computation
                    Shape: (1, batch size, decoder hidden size)
                attention: the attention weights computed on the current step of computation
                     Shape: (batch size, input length)

                    
        """
        # Problem 15: Replace the following with your own code. 
        embedded_input = self.dropout(self.embedding_layer(input)) # (batch size, embedding size)

        if self.use_attention:
            # previous_hidden: (1, batch_size, decoder hidden size)
            # Encoder outputs: (batch size, sequence length, encoder hidden size)
            # weighted_sum: (batch size, 1, decoder dimension)
            # Attention output: Shape: (batch size, 1, sequence length)

            weighted_sum, attention = self.attention(previous_hidden.squeeze(0), encoder_outputs)
            rnn_input = torch.cat((embedded_input.unsqueeze(1), weighted_sum), dim=2) # (batch size, 1, decoder + sequence length)
        else:
            rnn_input = embedded_input.unsqueeze(1)

        rnn_output, h = self.rnn(rnn_input, previous_hidden)
        # rnn_output: (batch size, 1, hidden_size)
        # h: (1, batch_size, hidden size)
        
        logits = self.linear(rnn_output.squeeze(1)) @ self.embedding_layer.weight.T

        if self.use_attention:
            attention = attention.squeeze(1)
        else:
            attention = None

        return logits, h, attention


class Seq2Seq(nn.Module):
    def __init__ (self, vocab_size: int, embedding_size: int, hidden_size: int, 
                  encoder_output_size: int, attention_size: int, recurrent_type: str = 'rnn', bidirectional: bool = True, maxlen: int = 10, dropout_prob: float = .5, use_attention: bool = True):
        """
        :param vocab_size: The size of the output vocabulary
        :param embedding_size: The size of the word embeddings used by
            the model
        :param hidden_size: The size of the encoder and decoder RNNs' hidden state 
        :param encoder_output_size: The size of the encoder embeddings
        :param attention_size: The size of the key and query vectors used by 
            the attention module
        :param recurrent_type: The kind of recurrent unit to be used ('rnn' or 
            'gru')
        :param bidirectional: If True, the encoder uses a bidirectional recurrent 
            network
        :param maxlen: The maximum number of symbols to be decoded
        :param: dropout_prob: The dropout probability
        :param: use_attention: If True, then the decoder uses the attention module for its computation
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.use_attention = use_attention
    
        self.encoder = RNNEncoder(vocab_size, embedding_size, hidden_size,  
                                  encoder_output_size,     
                                  recurrent_type = recurrent_type,
                                  bidirectional = bidirectional,
                                  dropout_prob = dropout_prob)
        self.attention = Attention(encoder_output_size, hidden_size, 
                                   attention_size)
        self.decoder = RNNDecoder(vocab_size, embedding_size, hidden_size, 
                                  attention = self.attention, 
                                  recurrent_type = recurrent_type,
                                  use_attention = use_attention,
                                  dropout_prob = dropout_prob)

    def forward(self, sentences: torch.LongTensor, targets: torch.LongTensor = None, 
                teacher_forcing_ratio: float = .5, maxlen: int = 0):
        """
        :param sentences: A batch of sentences, not including [BOS]
                but padded with [PAD], to be processed by the encoder.
            Shape: (batch size, sentence length)
        :param targets: A batch of target outputs corresponding to the inpur    
                sentences.
            Shape: (batch size, output length)
        :param teacher_forcing_ratio: the proportion of inputs to the decoder that 
                should be taken from the targets
        :param maxlen: the maximum length of the string to be decoded
        :return: result: A tensor containing the logits for each output 
                    position for each sentence in the batch
                Shape: (batch size, output length, vocab size)
                attention (optionally): A tensor containing the attention weights 
                    for each output position for each sentence in the batch.
                Shape: (batch size, output length, input length)
        """
        if maxlen == 0:
            maxlen = self.maxlen
        length = targets.shape[1] if targets != None else maxlen
        #Initialize tensors to contain the result and the attention weights
        result = torch.zeros(sentences.shape[0], length, 
                                 self.vocab_size)
        attention = torch.zeros(sentences.shape[0], length, sentences.shape[1])
        #Run the encoder on the input sentences
        encoder_outputs, prev_state = self.encoder(sentences)
        #Start off the decoder with input [BOS] (with index 0) for each 
        # sentence in the batch 
        input = torch.zeros(sentences.shape[0], dtype = int)
        #Do length-many steps of decoding and save the result
        for i in range(length):
            pred, prev_state, a = self.decoder(input, prev_state,
                                                encoder_outputs)
            result[:, i, :] = pred
            if self.use_attention:
                attention[:, i, :] = a
            #Choose the next input symbol for the decoder
            if random.random() > teacher_forcing_ratio or not self.training:
                input = pred.argmax(dim=-1)
            else:
                input = targets[:,i]                
        return result, attention
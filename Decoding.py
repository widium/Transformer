# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Decoding.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/09/19 11:12:36 by ebennace          #+#    #+#              #
#    Updated: 2022/10/03 11:31:41 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import tensorflow as tf
from tensorflow import Tensor
from keras.layers import Dense

from Multi_Head_Attention import Masked_Multi_Head_Attention_Layer, Multi_Head_Encoder_Attention_Layer
from keras.layers import Normalization


class  Decoding_Layer(tf.keras.layers.Layer):
    
    def __init__(self, dim : int, nbr_heads : int, mask : Tensor, **kwargs):
        
        self.dim = dim
        self.nbr_heads = nbr_heads
        self.mask = mask

        super(**kwargs).__init__()
       
    def build(self, input_shape):
        
        self.masked_multi_head_attention = Masked_Multi_Head_Attention_Layer(self.dim, self.nbr_heads, self.mask)
        self.encoder_multi_head_attention = Multi_Head_Encoder_Attention_Layer(self.dim, self.nbr_heads)
        self.normalization = Normalization()
        self.dense = Dense(256)
        
        super().build(input_shape)
    
    def call(self, x):
        
        output_embedding, encoder = x
        
        self_attention = self.masked_multi_head_attention(output_embedding)
        attention_normalize = self.normalization(self_attention + output_embedding)
        encoder_attention = self.encoder_multi_head_attention((attention_normalize, encoder, encoder))
        encoder_attention_normalize = self.normalization(attention_normalize + encoder_attention)
        dense = self.dense(encoder_attention_normalize)
        dense_normalize = self.normalization(encoder_attention_normalize + dense)
        
        return (dense_normalize) 
    
class Decoder_Layer(tf.keras.layers.Layer):
    
    def __init__(self, nb_decoder : int, dim : int, nbr_heads : int, mask : Tensor, **kwargs):
        
        self.nb_decoder = nb_decoder
        self.dim = dim
        self.nbr_heads = nbr_heads
        self.mask = mask
        
        super(**kwargs).__init__()
        
    def build(self, input_shape):
        
        self.decoder_layer_list = []

        for _ in range(self.nb_decoder):
            decoder_layer = Decoding_Layer(self.dim, self.nbr_heads, self.mask)
            self.decoder_layer_list.append(decoder_layer)
        
    
    def call(self, x):
        
        output_embedding, encoder = x
        
        decoder_output = output_embedding

        for decoder_layer in self.decoder_layer_list:
            decoder_output = decoder_layer((decoder_output, encoder))
        
        return (output_embedding);
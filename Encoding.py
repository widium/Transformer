# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Encoding.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/09/19 11:12:29 by ebennace          #+#    #+#              #
#    Updated: 2022/09/30 18:04:25 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import tensorflow as tf
from keras.layers import Dense

from Multi_Head_Attention import Multi_Head_Attention_Layer
from keras.layers import Normalization


class  Encoding_Layer(tf.keras.layers.Layer):
    
    def __init__(self, dim : int, nbr_heads : int, **kwargs):
        self.dim = dim
        self.nbr_heads = nbr_heads
        super(**kwargs).__init__()
       
    def build(self, input_shape):
        self.multi_head_attention = Multi_Head_Attention_Layer(self.dim, self.nbr_heads)
        self.normalization = Normalization()
        self.dense = Dense(256)
        super().build(input_shape)
    
    def call(self, x):
        
        attention = self.multi_head_attention(x)
        attention_normalize = self.normalization(attention + x)
        dense = self.dense(attention_normalize)
        output = self.normalization(dense + attention_normalize)
        
        return (output) 
    
class Encoder_Layer(tf.keras.layers.Layer):
    
    def __init__(self, nb_encoder : int, dim : int, nbr_heads : int, **kwargs):
        self.nb_encoder = nb_encoder
        self.dim = dim
        self.nbr_heads = nbr_heads
        super(**kwargs).__init__()
        
    def build(self, input_shape):
        
        self.encoding_layer_list = []

        for _ in range(self.nb_encoder):
            encoding_layer = Encoding_Layer(self.dim, self.nbr_heads)
            self.encoding_layer_list.append(encoding_layer)
    
    def call(self, x):
        
        for encoding_layer in self.encoding_layer_list:
            x = encoding_layer(x)
        return (x);
# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Encoding.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/10/06 14:35:58 by ebennace          #+#    #+#              #
#    Updated: 2022/10/06 14:37:21 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import tensorflow as tf
from keras.layers import Dense

from source.Attention.Multi.Multi_Head_Attention import Multi_Head_Attention_Layer
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
# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Decoding_layer.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/10/06 14:32:55 by ebennace          #+#    #+#              #
#    Updated: 2022/10/07 11:26:16 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import tensorflow as tf
from tensorflow import Tensor
from keras.layers import Dense

from ..Attention.Multi_Head.multi_masked import Masked_Multi_Head_Attention_Layer
from ..Attention.Multi_Head.multi_encoder_attention import Multi_Head_Encoder_Attention_Layer
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
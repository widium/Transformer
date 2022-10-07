# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Decoder.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/09/19 11:12:36 by ebennace          #+#    #+#              #
#    Updated: 2022/10/06 14:38:46 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import tensorflow as tf
from tensorflow import Tensor

from ..Decoding.Decoding_layer import Decoding_Layer

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
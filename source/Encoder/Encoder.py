# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Encoder.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/09/19 11:12:29 by ebennace          #+#    #+#              #
#    Updated: 2022/10/07 11:21:47 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import tensorflow as tf
from .Encoding import Encoding_Layer
    
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
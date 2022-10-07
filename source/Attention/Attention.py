# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Attention.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/09/19 11:09:12 by ebennace          #+#    #+#              #
#    Updated: 2022/10/06 14:30:58 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import tensorflow as tf
from keras.layers import Dense

from source.Attention.compute_attention.compute import compute_attention


class Attention_Layer(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super(**kwargs).__init__()
        
    def build(self, input_shape):
        self.Q_layer = Dense(256, name='query')
        self.K_layer = Dense(256, name='key')
        self.V_layer = Dense(256, name='value')
        super().build(input_shape)
    
    def call(self, input):
        
        Q = self.Q_layer(input)
        K = self.K_layer(input)
        V = self.V_layer(input)
        
        attention = compute_attention(Q, K, V)
        
        return (attention)
    
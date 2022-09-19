# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Embedding.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/09/19 11:05:04 by ebennace          #+#    #+#              #
#    Updated: 2022/09/19 11:05:37 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import tensorflow as tf
from keras.layers import Embedding

class  Embedding_Layer(tf.keras.layers.Layer):
    
    def __init__(self, nb_token, **kwargs):
        self.nb_token = nb_token
        super(**kwargs).__init__()
       
    def build(self, input_shape):
        self.word_embedding = Embedding(self.nb_token, 256)
        super().build(input_shape)
    
    def call(self, x):
        embedded = self.word_embedding(x)
        return (embedded) 
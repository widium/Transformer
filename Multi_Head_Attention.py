# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Multi_Head_Attention.py                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/09/19 11:07:00 by ebennace          #+#    #+#              #
#    Updated: 2022/10/02 11:36:48 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import tensorflow as tf
from keras.layers import Dense
from tensorflow import shape

from compute import compute_heads_dimensions
from mask import compute_masked_attention, create_mask
from tensor import duplicate_all_tensor
from compute import compute_attention
from compute import concatenate_attention_heads
from compute import create_multi_heads_attention_model


class Multi_Head_Attention_Layer(tf.keras.layers.Layer):
    
    def __init__(self, dim : int, nbr_heads : int, **kwargs):
        
        self.dim = dim
        self.nbr_heads = nbr_heads
        self.heads_dim = compute_heads_dimensions(dim, nbr_heads)
        
        super(**kwargs).__init__()
        
    def build(self, input_shape):
        
        self.Q_layer = Dense(self.dim, name='query')
        self.K_layer = Dense(self.dim, name='key')
        self.V_layer = Dense(self.dim, name='value')
        
        super().build(input_shape)
    
    def call(self, input):
        
        Q = self.Q_layer(input)
        K = self.K_layer(input)
        V = self.V_layer(input)
        
        batch_size = shape(Q)[0]
        sequence = shape(Q)[1]
        
        Qs, Ks, Vs = duplicate_all_tensor(Q, K, V, self.nbr_heads, self.heads_dim)
        Attention_heads = compute_attention(Qs, Ks, Vs)
        Attention_concatenate = concatenate_attention_heads(Attention_heads, self.dim, self.nbr_heads, self.heads_dim, batch_size, sequence)
        multi_heads_attention = create_multi_heads_attention_model(Attention_concatenate)
        
        return (multi_heads_attention)
    
    

class Masked_Multi_Head_Attention_Layer(tf.keras.layers.Layer):
    
    def __init__(self, dim : int, nbr_heads : int, mask_size : int, **kwargs):
        
        self.dim = dim
        self.nbr_heads = nbr_heads
        self.heads_dim = compute_heads_dimensions(dim, nbr_heads)
        self.mask = create_mask(mask_size)
        
        super(**kwargs).__init__()
        
    def build(self, input_shape):
        
        self.Q_layer = Dense(self.dim, name='query')
        self.K_layer = Dense(self.dim, name='key')
        self.V_layer = Dense(self.dim, name='value')
        
        super().build(input_shape)
    
    def call(self, input):
        
        Q = self.Q_layer(input)
        K = self.K_layer(input)
        V = self.V_layer(input)
        
        batch_size = shape(Q)[0]
        sequence = shape(Q)[1]
        
        Qs, Ks, Vs = duplicate_all_tensor(Q, K, V, self.nbr_heads, self.heads_dim)
        Attention_heads = compute_masked_attention(Qs, Ks, Vs, self.mask)
        Attention_concatenate = concatenate_attention_heads(Attention_heads, self.dim, self.nbr_heads, self.heads_dim, batch_size, sequence)
        multi_heads_attention = create_multi_heads_attention_model(Attention_concatenate)
        
        return (multi_heads_attention)
    
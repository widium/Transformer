# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    compute.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/09/19 11:07:24 by ebennace          #+#    #+#              #
#    Updated: 2022/09/19 11:19:59 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from tensorflow import Tensor
from keras.layers import Dense

from tensorflow.nn import softmax
from tensorflow.math import sqrt
from tensorflow import matmul
from tensorflow import reshape
from tensorflow import transpose
from tensorflow import shape

def create_attention_vector(Q : Tensor, K : Tensor):
    
    QK = matmul(Q, K, transpose_b=True)

    return (QK)

def normalize_vector(QK : Tensor):
    
    QK_N = QK / sqrt(256.)
    
    return (QK_N)

def create_vector_probability_attention(QK_N : Tensor):
    
    QK_S = softmax(QK_N, axis=-1)
    
    return (QK_S)

def add_attention_to_value(QK_S : Tensor, V : Tensor):
    
    attention = matmul(QK_S, V)
    
    return (attention)

def compute_attention(Q : Tensor, K : Tensor, V : Tensor):
    
    QK = create_attention_vector(Q, K)
    QK_N = normalize_vector(QK)
    QK_S = create_vector_probability_attention(QK_N)
    attention = add_attention_to_value(QK_S, V)
    
    return (attention)

def compute_heads_dimensions(representation : int, nbr_heads : int):
    
    heads_representation = representation // nbr_heads
    
    return (heads_representation) 

def duplicate_tensor(Tensor : Tensor, nbr_heads : int, heads_dim : int):
    
    batch_size = shape(Tensor)[0]
    sequence = shape(Tensor)[1]
    
    Tensor = reshape(Tensor, [batch_size, sequence, nbr_heads, heads_dim])
    Tensor = transpose(Tensor, [0, 2, 1, 3])
    Tensor = reshape(Tensor, [batch_size * nbr_heads, sequence, heads_dim])
    
    return (Tensor)

def duplicate_all_tensor(Q : Tensor, K : Tensor, V :Tensor, nbr_heads : int, heads_dim : int):
    
    Qs = duplicate_tensor(Q, nbr_heads, heads_dim)
    Ks = duplicate_tensor(K, nbr_heads, heads_dim)
    Vs = duplicate_tensor(V, nbr_heads, heads_dim)
    
    return (Qs, Ks, Vs)
    
def concatenate_attention_heads(Attention_heads : Tensor, representation : int, nbr_heads : int, heads_dim : int, batch_size : int, sequence : int):
    
    Attention_heads = reshape(Attention_heads, [batch_size, nbr_heads, sequence, heads_dim])
    Attention_heads = transpose(Attention_heads, [0, 2, 1, 3])
    Attention_concatenate = reshape(Attention_heads, [batch_size, sequence, nbr_heads * heads_dim])

    return (Attention_concatenate)

def create_multi_heads_attention_model(attention_concat : Tensor):
    
    multi_heads_attention = Dense(256, name="multi_heads_attention")(attention_concat)
    
    return (multi_heads_attention)
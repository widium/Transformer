# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    mask.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/10/02 11:22:45 by ebennace          #+#    #+#              #
#    Updated: 2022/10/03 07:50:05 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from compute import Tensor, tf
from compute import sequence_mask, cast, expand_dims
from compute import exp, reduce_sum
from compute import create_attention_vector, normalize_vector
from compute import create_vector_probability_attention, add_attention_to_value


def create_mask(size : int)-> Tensor:
    mask = sequence_mask(tf.range(size) + 1, size)
    mask = cast(mask, tf.float32)
    mask = expand_dims(mask, axis=0)
    return (mask)

def masked_softmax(X : Tensor , mask : Tensor):
    x_masked = X * mask
    t = exp(x_masked)
    t_masked = t * mask
    S = reduce_sum(t, axis=-1) 
    S_shape = expand_dims(S, axis=-1)
    D = t_masked / S_shape
    return (D)

def compute_masked_attention(Q : Tensor, K : Tensor, V : Tensor, mask : Tensor):
    
    QK = create_attention_vector(Q, K)
    QK_N = normalize_vector(QK)
    QK_S = masked_softmax(QK_N, mask)
    attention = add_attention_to_value(QK_S, V)
    
    return (attention)
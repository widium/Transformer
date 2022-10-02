# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    tensor.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/10/02 11:27:35 by ebennace          #+#    #+#              #
#    Updated: 2022/10/02 11:28:21 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from tensorflow import Tensor
from compute import shape, reshape, transpose

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
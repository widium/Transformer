# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Decoding.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/09/19 11:12:36 by ebennace          #+#    #+#              #
#    Updated: 2022/10/02 11:45:44 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import tensorflow as tf
from keras.layers import Dense

from Multi_Head_Attention import Masked_Multi_Head_Attention_Layer
from keras.layers import Normalization



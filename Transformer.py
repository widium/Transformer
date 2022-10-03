# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Transformer.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/10/03 11:34:39 by ebennace          #+#    #+#              #
#    Updated: 2022/10/03 11:38:03 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from keras.layers import Input
from mask import create_mask
from Encoding import Encoder_Layer
from Decoding import Decoder_Layer
from Embedding import Embedding_Layer
from keras import Model
from keras.layers import Dense
from tensorflow.nn import softmax
from tensorflow import Tensor

NBR_TOKEN = 5
INPUT_SIZE = 5
OUTPUT_SIZE = 6
MASK_SIZE = 6

def Transformer(coding_block : int, dim : int, nbr_heads : int):
    
    mask = create_mask(MASK_SIZE)
    
    input_token = Input(shape=(INPUT_SIZE))
    output_token = Input(shape=(OUTPUT_SIZE))
    
    input_embedding = Embedding_Layer(INPUT_SIZE)(input_token)
    output_embedding = Embedding_Layer(OUTPUT_SIZE)(output_token)
    
    encoder = Encoder_Layer(coding_block, dim, nbr_heads)(input_embedding)
    decoder = Decoder_Layer(coding_block, dim, nbr_heads, mask)((output_embedding, encoder))

    
    model = Model([input_token, output_token], decoder)
    model.summary()
    return (model)
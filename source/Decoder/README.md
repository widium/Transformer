# Decoder

![](https://i.imgur.com/TFIaC1G.png)

## [Class Decoding_layer()](Decoding_layer.py)
#### Make the Scaled Dot Product Attention on : 
- *the Output Embedding*
- *the Output of the Encoder Layer in relation with the Attention created on the Output Embedding*.

![](https://i.imgur.com/I5ge44l.png)
***
- we **create** a [Masked Multi-head Attention](../Attention/Multi_Head/multi_masked.py) to make attention on the already predicted tokens 
- Then we **create** a [Multi_head_Encoder_Attention_layer](../Attention/Multi_Head/multi_encoder_attention.py) to create attention on the [Class Encoder_Layer](../Encoder/) in relation to the attention create **on the already predicted tokens** 
- We **create** a final projection 
- As for the [Class Encoder_Layer](../Encoder/) we *Normalize* after each [Multi Head Attention](../Attention/Multi_Head/) and [projection](../Attention/README.md#query-projection).

~~~~C
self_attention = masked_multi_head_attention(output_embedding)
attention_normalize = normalization(self_attention + output_embedding)
encoder_attention = encoder_multi_head_attention((attention_normalize, encoder, encoder))
encoder_attention_normalize = normalization(attention_normalize + encoder_attention)
dense = dense(encoder_attention_normalize)
dene_normalize = normalization(encoder_attention_normalize + dense)
~~~~

## [Class Decoder_layer()](Decoder.py)
is to call $\large N$ times the [Class Decoding_layer()](Decoding_layer.py) to retrieve :
- to get a different semantic representation for each decoding 
- which will allow to have **a diversity of the representation of the information** to better predict the Output
~~~C
decoder = Decoder_Layer(coding_block, dim, nbr_heads, mask)((output_embedding, encoder))
~~~
~~~python
for decoder_layer in self.decoder_layer_list:
            decoder_output = decoder_layer((decoder_output, encoder))
~~~
![](https://i.imgur.com/sn6UUrJ.png)

### Create Layer  To call N times [Class Decoding_layer()](Decoding_layer.py)
- **Argument** : 
	- `dim`: the *representation size* of each token
	- `nbr_heads` : *number of Attention Head]]*
	- `nb_decoder`: number of desired [Class Decoding_layer()](Decoding_layer.py)
- __init__() :
	- *assign* `dim` and `nbr_heads` and `nb_decoder` to the class
- **Define the same Construction - Layer Custom build()** :
	- *Initialize* a Python Lists of [Class Decoding_layer()](Decoding_layer.py) with the `nb_encoder`
- **Define Layer Calculation - Layer Custom call()** :
	- *Recover* **the output of the [Class Encoder_layer()](../Encoder/) and the output_embedding**
	- *define* the output of the decoder 
	- *Encode* loop the input with each [Class Decoding_layer()](Decoding_layer.py)
	- *return* the output of the last [Class Decoding_layer()](Decoding_layer.py)
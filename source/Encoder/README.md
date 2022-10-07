## Encoder

![](https://i.imgur.com/8DVleQT.png)

### The [Class Encoding_layer()](Encoding.py)
allows to encode the **Embedding** input with :
- [Multi Head Attention](../Attention/Multi_Head/)
- Layer_Normalization of attention

it allows to learn the Specialization of Attention with its [Multi Head Attention](../Attention/Multi_Head/) in order to encode the sequence in the best way

~~~python
attention = multi_head_attention(x)
attention_normalize = normalization(attention + x)
dense = dense(attention_normalize)
output = normalization(dense + attention_normalize)
~~~

### The [Class Encoder_Layer()](Encoder.py)

~~~python
for encoding_layer in self.encoding_layer_list:
    x = encoding_layer(x)
return (x);
~~~

- The Encoder calls **N** times the [Class Encoding_layer()](Encoder.py) to get several semantic representations of the same **Sequence** which allows to capture a **semantic diversity** of the same input ! and give after to the [Decoder Layer](../Decoder/)

- Each [Class Encoding_layer()](Encoding.py) **receives in input the output of the previous one** 
- it **re-encodes** each previous [Class Encoding_layer()](Encoding.py) which allows to get an optimal Encoding Data !


![](https://i.imgur.com/8tj3aYI.png)

## Normalize Layer
- Residual Layer Normalisation after [Multi_Head_Attention](#multi-head-attention)
~~~python
from keras.layers import Normalization

normalization = Normalization(axis=None)(multi_head_attention_layer + embedding_layer)
~~~
- This layer allows to Normalize the [Multi_Head_Attention](#multi-head-attention) to avoid **Vanishing & Exploding Gradients** Problem

![](https://i.imgur.com/fydPCTR.png)
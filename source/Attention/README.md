# Attention
- [1. Attention Mecanism](#attention-mecanism)
- [2. Scaled Dot Product Attention ](#scaled-dot-product-attention )
- [3. Query Projection](#query-projection)
- [4. Compute Attention](#compute-attention)
- [5. Self Attention](#self-attention)
- [6. Multi Head Attention](Multi_Head/)

## Attention Mecanism
Contrary to Multi-Layer-Perceptron or CNN, the Attention mechanism allows a dynamic connection between 2 Layers,  it allows to choose which information will be sent to the next layer !
the model learn on which information it should focus its attention.

![](https://i.imgur.com/eL8ptdI.png)
## Scaled Dot Product Attention 

To create Attention we need to create 3 [Projection](#query-projection) of the Input Tensor called **$\large Q$ $\large K$ $\large V$** and [Compute Attention](compute.py)

![](https://i.imgur.com/Sq2oXr2.png)
***
### Query Projection

- $\LARGE Q$ **= Importance** of the information
- $\LARGE K$ **= List** of the information
- $\LARGE V$ **= Value** of the information

*$\large Q$* asks *$\large K$* if he has any important information to give him, if yes we get his value with *$\large V$*

![](https://i.imgur.com/Hypsu3O.png)

These 3 Tensors have **the ability to learn to retrieve the best information** to maximize the general Attention, they will be defined as 
- 3 Dense Models
- which will take as input a Tensor.

~~~python
Q = Dense(256, name= 'query')(Tensor)
K = Dense(256, name= 'key')(Tensor)
V = Dense(256, name= 'value')(Tensor)
~~~
### Compute Attention

~~~python
def compute_attention(Q : Tensor, K : Tensor, V : Tensor):
    
    QK = create_attention_vector(Q, K)
    QK_N = normalize_vector(QK)
    QK_S = create_vector_probability_attention(QK_N)
    attention = add_attention_to_value(QK_S, V)
    
    return (attention)
~~~
![](https://i.imgur.com/Sq2oXr2.png)

~~~python
from keras.layers import Dense
from tensorflow import Tensor
from tensorflow.nn import softmax
from tensorflow.math import sqrt
from tensorflow import matmul
~~~
#### *1. Recuperation de L'input Tensor*
~~~python
input = tf.random.uniform((1, 5, 256))

(1, 5, 256)
~~~
#### *2. Initialiser les 3 Model de Projection de Tensor*
~~~python
Q = Dense(256, name= 'query')(input)
K = Dense(256, name= 'key')(input)
V = Dense(256, name= 'value')(input)


print(f"Q : {Q.shape}")
print(f"K : {K.shape}")
print(f"V : {V.shape}")

Q : (1, 1, 256)
K : (1, 5, 256)
V : (1, 5, 256)
~~~
### *3. Compute Attention*
~~~python
attention = compute_attention(Q, K, V)
~~~

~~~python
(1, 5, 256) tf.Tensor(
[ 0.00851879  0.00807548 -0.02141227 ... -0.00488998 -0.02339829
    0.01425959]
  [ 0.008518    0.00807689 -0.02140853 ... -0.00489697 -0.02340104
    0.01425432]
  [ 0.00852995  0.00806922 -0.02142657 ... -0.00489593 -0.0234261
    0.01429098]
  [ 0.00852115  0.00808086 -0.02140692 ... -0.00489073 -0.02340939
    0.01425311]
  [ 0.00852582  0.00807407 -0.02141737 ... -0.00488885 -0.02341493
    0.0142769 ], shape=(1, 5, 256), dtype=float32)
~~~

### Self Attention
- the self attention allows to create attention on all the tokens of the sequence at the same time !
- and so each token will also do *attention on itself.*
- Each token will then be able to *fetch information on the other tokens* of the sequence 

- A query $\large Q$ will be created for each token of the sequence, this process is called **Multi-head Attention**.

![](https://i.imgur.com/PgGGIk7.png)



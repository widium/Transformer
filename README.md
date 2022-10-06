# Transformer
### **Create Transformer from scratch,  from the paper "Attention is all you need".**

- [1. Understanding Architecture](#understanding-architecture)
- [2. Attention Mecanism](#attention-mecanism)
    - [1. Scaled Dot Product Attention ](#scaled-dot-product-attention )
        - [1. Query Projection](#query-projection)
        - [2. Compute Attention](#compute-attention)
    - [2. Self Attention](#self-attention)
    - [3. Multi Head Attention](#multi-head-attention)
- [2. Embedding](#embedding)
- [3. Positional Encoding](#positional-encoding)
- [4. Encoder](#encoder)
- [5. Decoder](#decoder)

## Understanding Architecture
Transformer use the [Attention Mecanism](#attention-mecanism) for learn to extract features with [Encoder](#encoder) and [Decoder](#decoder)
![](https://i.imgur.com/dggyZEz.png)

## Attention Mecanism
Contrary to Multi-Layer-Perceptron or CNN, the Attention mechanism allows a dynamic connection between 2 Layers,  it allows to choose which information will be sent to the next layer !
the model learn on which information it should focus its attention.
![](https://i.imgur.com/eL8ptdI.png)
### Scaled Dot Product Attention 

To create Attention we need to create 3 Projections of the Input Tensor called **$\large Q$ $\large K$ $\large V$**

#### Query Projection

- *$\LARGE Q = $* **Importance** of the information
- *$\LARGE K = $* **List** of the information
- *$\LARGE V = $* **Value** of the information

*$\large Q$* asks *$\large K$* if he has any important information to give him, if yes we get his value with *$\large V$*

![](https://i.imgur.com/Hypsu3O.png)

These 3 Tensors have **the ability to learn to retrieve the best information** to maximize the general Attention, they will be defined as 
- 3 Multivariate MLP Models
- which will take as input a Tensor.

~~~python
Q = Dense(256, name= 'query')(Tensor)
K = Dense(256, name= 'key')(Tensor)
V = Dense(256, name= 'value')(Tensor)
~~~
***
#### Compute Attention
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
encodeur_state = tf.random.uniform((1, 5, 256))
decodeur_state = tf.random.uniform((1, 1, 256))

print(encodeur_state.shape)
print(decodeur_state.shape)

(1, 5, 256)
(1, 1, 256)
~~~
#### *2. Initialiser les 3 Model de Projection de Tensor*
~~~python
Q = Dense(256, name= 'query')(decodeur_state)
K = Dense(256, name= 'key')(encodeur_state)
V = Dense(256, name= 'value')(encodeur_state)


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
[[[ 0.00851879  0.00807548 -0.02141227 ... -0.00488998 -0.02339829
    0.01425959]
  [ 0.008518    0.00807689 -0.02140853 ... -0.00489697 -0.02340104
    0.01425432]
  [ 0.00852995  0.00806922 -0.02142657 ... -0.00489593 -0.0234261
    0.01429098]
  [ 0.00852115  0.00808086 -0.02140692 ... -0.00489073 -0.02340939
    0.01425311]
  [ 0.00852582  0.00807407 -0.02141737 ... -0.00488885 -0.02341493
    0.0142769 ]]], shape=(1, 5, 256), dtype=float32)
~~~

### Self Attention
- la self attention permet de **creer de l'attention sur tous les token de la sequence en meme temps !**
- et donc chaque token fera egalement de *l'attention sur lui meme.*
- Chaque token pourra alors *recuperer des informations sur les autre token* de la sequence 

- Une query $\large Q$ sera creer pour chaque token de la sequence !, ce proceder s'appelle de la **Multi-head Attention**

![](https://i.imgur.com/PgGGIk7.png)

### Multi-head Attention
Create Multi-Sub-Query and Multi [Attention Head](#scaled-dot-product-attention )
- ### [Class Multi_Head_Attention_Layer](source/Multi_Head_Attention.py)
### **$$Multihead(Q, W, K) = Concat(head_1, head_2, ..., head_h)W^O$$**
### **$$\text {where head}_i = Attention(QW^Q_i, KW^K_i, VW^V_i)$$**
![](https://i.imgur.com/rWszwyg.png)

### Attention Specialization

- If we create several Queries for the same Token we could have **Queries that specialize in information retrieval** and thus maximize the attention of all tokens
- The Number of [Attention Head](#scaled-dot-product-attention ) is thus a [[Hyperparameter]] to allow to manage the Sequence of big dimensions !

![](https://i.imgur.com/ysBoWSN.png)

### *Concatenate and learn specialization :*
- We define a number [Attention Head](#scaled-dot-product-attention) by token 
- Each **Head** *will* specialize in an information search
- Concatenate all the Attention for each token
- We add a MLP Layer to **learn to specialize the Attention**.

![](https://i.imgur.com/UqnK0Ir.png)

## Embedding
Soon operational
## Positional Encoding
Soon operational
## Encoder

## Decoder
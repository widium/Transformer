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

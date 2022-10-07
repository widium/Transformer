# Multi-head Attention
### [1. Attention Specialization](#attention-specialization)
### [2. Concatenate specialization](#concatenate-specialization)
### [3. Duplicate Tensor](#duplicate-tensor)
### [4. Concatenate Attention](#concatenate-attention)
### [5. Mask](#mask)
## Class
### [Class Multi_Head_Attention_Layer](#multiheadattentionlayer)
### [Class Masked_Multi_Head_Attention_Layer](#maskedmultiheadattentionlayer)
### [Class Multi_Head_Encoder_Attention_Layer](#multiheadencoderattentionlayer)

***

Create Multi-Sub-Query and Multi [Attention Head](#scaled-dot-product-attention )

### **$$Multihead(Q, W, K) = Concat(head_1, head_2, ..., head_h)W^O$$**
### **$$\text {where head}_i = Attention(QW^Q_i, KW^K_i, VW^V_i)$$**
![](https://i.imgur.com/rWszwyg.png)

## Attention Specialization

- If we create several Queries for the same Token we could have **Queries that specialize in information retrieval** and thus maximize the attention of all tokens
- The Number of [Attention Head](#scaled-dot-product-attention ) is thus a Hyperparameter to allow to manage the Sequence of big dimensions !

![](https://i.imgur.com/ysBoWSN.png)

## Concatenate specialization
- We define a number [Attention Head](#scaled-dot-product-attention) by token 
- Each **Head** *will* specialize in an information search
- Concatenate all the Attention for each token
- We add a Dense Layer to **learn to specialize the Attention**.

![](https://i.imgur.com/RMM4Qrt.png)

### Compute the representation size of the Tensor sub-models.
- Each [Tensor Projection](#query-projection) has a size of :
	- **`(Batch_size, sequence, representation)`**
- A representation of a basic token is `256`. 
- If we want to **duplicate the current [Tensor Projection](#query-projection) in 8** under [Tensor Projection](#query-projection)
- We can then **simply divide the representation by the number of sub [Tensor Projection](#query-projection) we want**.
	- **`dim = 256 // 8`**
- in this example it will give us **a representation of 32** for each sub [Tensor Projection](#query-projection)

~~~python
def compute_heads_dimensions(representation : int, nbr_heads : int):
    
    heads_representation = representation // nbr_heads
    
    return (heads_representation) 
~~~
~~~python
compute_heads_dimensions(256, 8)

>> 32
~~~
![](https://i.imgur.com/UqnK0Ir.png)
### duplicate tensor

![](https://i.imgur.com/z7rafpp.png)
- Tensor Original `(batch, sequence, representation)` `(1, 5, 256)`
~~~python
Tensor shape : (1, 5, 256)
~~~
### **Split** into `nbr_heads=8`
- small Tensor with a reshape compute upstream with Compute_heads_dim() `heads_dim=32`
- Reshape - Numpy|Reshape() the Tensor in : `[batch_size, sequence, nbr_heads, heads_dim]`
~~~python
Tensor shape : (1, 5, 8, 32)
~~~
### **Transpose** the `sequence` with the `nbr_heads` :
- *1* times *8* Attention Head for *5* sequence elements with a representation of *32*.
- To define the correct dimension calculation be :
`1 x 8 x 5 x 32 = 1280` Element in the Tensor
- We then transpose the elements with their dimension index 
~~~python
transpose(Tensor, [0, 2, 1, 3])

Tensor shape : (1, 8, 5, 32)
~~~

#### **Reshape** in 3D
- To be able to Compute Attention in 3D **just remove the first dimension** **by multiplying** it with the `nbr_heads`:

~~~python
reshape(Tensor, [batch_size * nbr_heads, sequence, heads_dim])

Tensor shape : (8, 5, 32)
~~~
## Concatenate Attention
![](https://i.imgur.com/diMrRjl.png)

 **Reshape** by **adding** a Batch-Size dimension
~~~python
Attention_heads = reshape(Attention_heads, [batch_size, nbr_heads, sequence, heads_dim])

Attention_heads (1, 8, 5, 32)
~~~
 **Transpose** the heads
- Transpose** so as to **multiply the number of heads with their representation**. 
`(1, 8, 5, 32)` -> `(1, 5, 8, 32)`
`8 x 32 = 256`
~~~python
Attention_heads = transpose(Attention_heads, [0, 2, 1, 3])

Attention_heads (1, 5, 8, 32)
~~~

 **Reshape** in 3D
- Multiply the Number of heads with their representation
`8 x 32 = 256`
~~~python
Attention_concatenate = reshape(Attention_heads,
         [batch_size, sequence, nbr_heads * heads_dim])

Attention_concatenate (1, 5, 256)
~~~

## [Class Attention_Layer()](Attention.py)
Create Layer of Keras With [3 Tensor Projection Model](../../README.md#query-projection)
- Define  3 Tensor with a **representation** of `256` in this case
- Calculate the attention with [compute_attention()](../../README.md#scaled-dot-product-attention)

# Mask
## Create mask
~~~python
X = np.array([[[9, 32, 71, 54, 4]]])
'shape (1, 1, 5)'

mask = tf.sequence_mask(tf.range(5) + 1, 5)
mask = tf.cast(mask, tf.float32)
mask = tf.expand_dims(mask, axis=0)

output = X * mask

<tf.Tensor: 'shape=(1, 5, 5)',
			dtype=float32, 
			numpy= array(
	  [[[ 9., 0., 0., 0., 0.],
        [ 9., 32., 0., 0., 0.],
        [ 9., 32., 71., 0., 0.],
        [ 9., 32., 71., 54., 0.],
        [ 9., 32., 71., 54., 4.]], dtype=float32)>
~~~

### Create a Boolean mask

- With the function [tf.sequence_mask](https://www.tensorflow.org/api_docs/python/tf/sequence_mask) we **apply a boolean mask on the indexes** we want to hide 
- Define a sequence size `SEQUENCE_SIZE`.
- Define a ==step of mask `+ 1, SEQUENCE_SIZE` in this example a step from `1` to 5
- Set the sequence size and **apply** the mask
~~~python
SEQUENCE_SIZE = 5

mask = tf.sequence_mask(tf.range(SEQUENCE_SIZE) + 1, SEQUENCE_SIZE)
print(mask)

tf.Tensor(
[['True' False False False]
 ['True' 'True' False False False]
 ['True' 'True' 'True' False False]
 ['True' 'True' 'True' 'True' False]
 ['True' 'True' 'True' 'True' 'True']], shape=(5, 5), dtype=bool)
~~~

### Convert a Boolean mask to Float

- you have to cast the boolean mask to apply it on another tensor
- Convert the boolean mask to floats
~~~python
mask = tf.sequence_mask(tf.range(5) + 1, 5)
mask = tf.cast(mask, tf.float32)


tf.Tensor(
[[1. 0. 0. 0. 0.]
 [1. 1. 0. 0. 0.]
 [1. 1. 1. 0. 0.]
 [1. 1. 1. 1. 0.]
 [1. 1. 1. 1. 1.]], shape=(5, 5), dtype=float32)
~~~

### Add a dimension to the Mask

- pass from the shape of a Matrix to the shape of a Tensor
~~~python
mask = tf.sequence_mask(tf.range(5) + 1, 5)
mask = tf.cast(mask, tf.float32)
mask = tf.expand_dims(mask, axis=0)

tf.Tensor(
[[[1. 0. 0. 0. 0.]
  [1. 1. 0. 0. 0.]
  [1. 1. 1. 0. 0.]
  [1. 1. 1. 1. 0.]
  [1. 1. 1. 1. 1.]], shape=(1, 5, 5), dtype=float32)
~~~


- after having **created the mask**
- **multiply** the 2 Tensor 
- will give 5 sequence with the application of the progressive mask
~~~python
X = np.array([[[9, 32, 71, 54, 4]]])
'shape (1, 1, 5)'

mask = tf.sequence_mask(tf.range(5) + 1, 5)
mask = tf.cast(mask, tf.float32)
mask = tf.expand_dims(mask, axis=0)

output = X * mask

<tf.Tensor: 'shape=(1, 5, 5)',
			dtype=float32, 
			numpy= array(
	  [[[ 9., 0., 0., 0., 0.],
        [ 9., 32., 0., 0., 0.],
        [ 9., 32., 71., 0., 0.],
        [ 9., 32., 71., 54., 0.],
        [ 9., 32., 71., 54., 4.]], dtype=float32)>
~~~
## Masked Softmax 

### *Mask the Values* that **should not be taken into account** by the calculation
- Create a sequence mask
- apply the mask with a multiplication on :
	- The X input** to mask the desired Indexes
	- after the calculation of the Exponential `t`` 
~~~python
masked_x = x * mask
masked_t = exp(masked_x) * mask
~~~
- Compute the softmax normally :
~~~python
def masked_softmax(X : Tensor , mask : Tensor) :
    x_masked = X * mask
    t = tf.math.exp(x_masked)
    t_masked = t * mask
    S = tf.reduce_sum(t, axis=-1) 
    S_shape = tf.expand_dims(S, axis=-1)
    D = t_masked / S_shape
    return (D)
~~~

### *Result* 
- The `Masked_softmax` contains 5 sequences with the mask removed as you go along
~~~python
softmax = softmax(X, axis=-1)
masked_softmax = masked_softmax(X, mask)
~~~
~~~python
"softmax classic"
tf.Tensor(
[[[1.18e-27 1.15e-17 9.99e-01 4.13e-08 7.98e-30]], 
	shape=(1, 1, 5), dtype=float64)

"Masked Softmax"
<tf.Tensor: shape=(1, 5, 5), dtype=float32, numpy=
array([[1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00],
        [1.18e-27, 1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00],
        [1.18e-27, 1.15e-17, 1.00e+00, 0.00e+00, 0.00e+00],
        [1.18e-27, 1.15e-17, 1.00e+00, 4.13e-08, 0.00e+00],
        [1.18e-27, 1.15e-17, 1.00e+00, 4.13e-08, 7.98e-30]],
		dtype=float32)>
~~~
# Multi Head Attention

## Multi_Head_Attention_Layer()
[Class Multi_Head_Attention_Layer()](Multi_Head_Attention.py)
- **Argument** : 
	- `dim`: the *representation size* of each token
	- `nbr_heads` : *number of [Attention Head](../compute_attention/compute.py)*
- __init__() :
	- *assign* `dim` and `nbr_heads` to the class
	- *create* `heads_dim` with [Compute_heads_dimensions()](../compute_attention/compute.py)
- **Define the same Construction** :
	- *Create* 3 [Tensor](../tensor_projection/tensor.py) for the 3 queries 
- **Define Layer Calculation** :

	- *Fetch* the 3 [Tensor](../tensor_projection/tensor.py)
	- *retrieve* the Batch-Size
	- *recover* the sequence size
	- [Duplicate_all_Tensor()](../tensor_projection/tensor.py)
	- *Create* the N [Attention Head](../compute_attention/compute.py) with [compute_all_attention_heads()](../compute_attention/compute.py)
	- *Assemble* the Heads with [Concatenate_attention_heads()](../compute_attention/compute.py)
	- *Create* projection 
	- *Return* the representation of the sequence
## Masked_Multi_Head_Attention_Layer()
[Class Masked_Multi_Head_Attention_Layer()](multi_masked.py)
### Change the Computation of the Attention Head
- The only difference with the Multi_Head_Attention Layer is that we add a mask to calculate the attention

- With the help of a Masked Softmax we will [Compute_masked_attention_heads()](mask.py) for each query by **masking the future tokens** :

## Multi_Head_Encoder_Attention_Layer()
![](https://i.imgur.com/KN2IotV.png)

Same operation as a [Class Multi_Head_Attention_Layer()](Multi_Head_Attention.py) except that we *can define the Tensor Projection Model* to retrieve different Tensor Query

### Multi_input :
~~~C
"Classic"
attention = self.multi_head_attention(x)

"Multi"
attention = self.multi_head_attention((out, encoder, encoder))
~~~
The Tensor Projection will not be on the same input
~~~C
"Classic"
attention = self.multi_head_attention(x)
    
	Q = self.Q_layer(input)
	K = self.K_layer(input)
	V = self.V_layer(input)

"Multi"
query, key, value = input
        
	Q = self.Q_layer(query)
	K = self.K_layer(key)
	V = self.V_layer(value)
~~~

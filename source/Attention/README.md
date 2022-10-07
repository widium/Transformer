# Attention
- [1. Attention Mecanism](#attention-mecanism)
- [2. Scaled Dot Product Attention ](#scaled-dot-product-attention )
	- [1. Query Projection](tensor_projection/tensor.py)
	- [2. Compute Attention](#compute-attention)
- [3. Self Attention](#self-attention)
- [4. Multi Head Attention](#multi-head-attention)

## Attention Mecanism
Contrary to Multi-Layer-Perceptron or CNN, the Attention mechanism allows a dynamic connection between 2 Layers,  it allows to choose which information will be sent to the next layer !
the model learn on which information it should focus its attention.

![](https://i.imgur.com/eL8ptdI.png)
## Scaled Dot Product Attention 

To create Attention we need to create 3 [Projection](tensor_projection/tensor.py) of the Input Tensor called **$\large Q$ $\large K$ $\large V$** and [Compute Attention](compute_attention/compute.py)

![](https://i.imgur.com/Sq2oXr2.png)
***
### Self Attention
- la self attention permet de **creer de l'attention sur tous les token de la sequence en meme temps !**
- et donc chaque token fera egalement de *l'attention sur lui meme.*
- Chaque token pourra alors *recuperer des informations sur les autre token* de la sequence 

- Une query $\large Q$ sera creer pour chaque token de la sequence !, ce proceder s'appelle de la **Multi-head Attention**

![](https://i.imgur.com/PgGGIk7.png)

### Multi-head Attention
Create Multi-Sub-Query and Multi [Attention Head](#scaled-dot-product-attention )
- ### [Class Multi_Head_Attention_Layer](source/Attention/Multi_Head_Attention.py)
### **$$Multihead(Q, W, K) = Concat(head_1, head_2, ..., head_h)W^O$$**
### **$$\text {where head}_i = Attention(QW^Q_i, KW^K_i, VW^V_i)$$**
![](https://i.imgur.com/rWszwyg.png)

### Attention Specialization

- If we create several Queries for the same Token we could have **Queries that specialize in information retrieval** and thus maximize the attention of all tokens
- The Number of [Attention Head](#scaled-dot-product-attention ) is thus a Hyperparameter to allow to manage the Sequence of big dimensions !

![](https://i.imgur.com/ysBoWSN.png)

### *Concatenate and learn specialization :*
- We define a number [Attention Head](#scaled-dot-product-attention) by token 
- Each **Head** *will* specialize in an information search
- Concatenate all the Attention for each token
- We add a MLP Layer to **learn to specialize the Attention**.

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

# Multi Head Attention

## [Class Multi_Head_Attention_Layer()](Multi_Head/Multi_Head_Attention.py)
## [Class Masked_Multi_Head_Attention_Layer()](Multi_Masked/multi_masked.py)
## [Class Multi_Head_Encoder_Attention_Layer()](Multi_Encoder_Attention/multi_encoder_attention.py)




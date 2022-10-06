# Attention

## [Class Attention_Layer()](Attention.py)
Create Layer of Keras With [3 Tensor Projection Model](../../README.md#query-projection)
- Define  3 Tensor with a **representation** of `256` in this case
- Calculate the attention with [compute_attention()](../../README.md#scaled-dot-product-attention)

# Multi Head Attention

## [Class Multi_Head_Attention_Layer()](Multi_Head_Attention.py)
- **Argument** : 
	- `dim`: the *representation size* of each token
	- `nbr_heads` : *number of [Attention Head](../../README.md#scaled-dot-product-attention)*
- __init__() :
	- *assign* `dim` and `nbr_heads` to the class
	- *create* `heads_dim` with [Compute_heads_dimensions()](../../README.md#compute-the-representation-size-of-the-tensor-sub-models)
- **Define the same Construction** :
	- *Create* 3 [Tensor](../../README.md#query-projection) for the 3 queries 
- **Define Layer Calculation** :

	- *Fetch* the 3 [Tensor](../../README.md#query-projection)
	- *retrieve* the Batch-Size
	- *recover* the sequence size
	- [Duplicate_all_Tensor()](../../README.md#duplicate-tensor)
	- *Create* the N [Attention Head](../../README.md#scaled-dot-product-attention) with [compute_all_attention_heads()](../../README.md#compute-attention)
	- *Assemble* the Heads with [Concatenate_attention_heads()](../../README.md#concatenate-attention)
	- *Create* projection 
	- *Return* the representation of the sequence
## [Class Masked_Multi_Head_Attention_Layer()](Multi_Head_Attention.py) && [Class Multi_Head_Encoder_Attention_Layer()](Multi_Head_Attention.py)

### [Class Masked_Multi_Head_Attention_Layer()](Multi_Head_Attention.py)
 Change the Computation of the Attention Head
- With the Masked Softmax we will [[Compute_masked_attention_heads()]] for each query by **masking the future tokens** :

### *The only difference with the Multi_Head_Attention
- ##### *adding a mask in parameters* 
- ##### *a different attention calculation*

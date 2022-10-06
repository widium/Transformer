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
- Define** a sequence size `SEQUENCE_SIZE`.
- Define** a ==step of mask `+ 1, SEQUENCE_SIZE` in this example a step from `1` to 5
- Set** the sequence size and **apply** the mask
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
- Convert** the boolean mask to floats
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
- will give 5 sequence** with the application of the progressive mask
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
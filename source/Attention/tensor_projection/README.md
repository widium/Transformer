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
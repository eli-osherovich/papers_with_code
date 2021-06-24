An implementation of
[Meta Decision Trees for Explainable Recommendation Systems](https://arxiv.org/abs/1912.09140)


The implementation follows the paper pretty closely. The main difference are:
* Outputs are scaled to the interval [0, 1] (one must scale RMSE back to get proper result)
* Added a penalty for non-even split at internal nodes. It should promote splits that are more 50-50

**Note**: For some reason, `tf-eigen` is several times faster then `tf-mkl` on this problem.

Simple demo:

```
./demo.py
```

It should produce something along this output:

```
Epoch 1/10000
4/4 [==============================] - 4s 45ms/step - loss: 0.2221 - mae: 0.1944 - root_mean_squared_error: 0.2484
Epoch 2/10000
4/4 [==============================] - 0s 43ms/step - loss: 0.2095 - mae: 0.1758 - root_mean_squared_error: 0.2216
Epoch 3/10000
4/4 [==============================] - 0s 40ms/step - loss: 0.2060 - mae: 0.1649 - root_mean_squared_error: 0.2135
Epoch 4/10000
4/4 [==============================] - 0s 39ms/step - loss: 0.2023 - mae: 0.1575 - root_mean_squared_error: 0.2046
Epoch 5/10000
4/4 [==============================] - 0s 39ms/step - loss: 0.2048 - mae: 0.1613 - root_mean_squared_error: 0.2108
Epoch 6/10000
4/4 [==============================] - 0s 40ms/step - loss: 0.2039 - mae: 0.1629 - root_mean_squared_error: 0.2085
Epoch 7/10000
4/4 [==============================] - 0s 40ms/step - loss: 0.2028 - mae: 0.1568 - root_mean_squared_error: 0.2059
Epoch 8/10000
4/4 [==============================] - 0s 39ms/step - loss: 0.2057 - mae: 0.1660 - root_mean_squared_error: 0.2128
Epoch 9/10000
4/4 [==============================] - 0s 39ms/step - loss: 0.2005 - mae: 0.1562 - root_mean_squared_error: 0.2003
Epoch 10/10000
4/4 [==============================] - 0s 38ms/step - loss: 0.2027 - mae: 0.1588 - root_mean_squared_error: 0.2057
 .
 .
 .
Epoch 2145/10000
4/4 [==============================] - 0s 40ms/step - loss: 0.1769 - mae: 0.0820 - root_mean_squared_error: 0.1194
Epoch 2146/10000
4/4 [==============================] - 0s 40ms/step - loss: 0.1756 - mae: 0.0812 - root_mean_squared_error: 0.1140
Epoch 2147/10000
4/4 [==============================] - 0s 47ms/step - loss: 0.1751 - mae: 0.0808 - root_mean_squared_error: 0.1129
Epoch 2148/10000
4/4 [==============================] - 0s 44ms/step - loss: 0.1749 - mae: 0.0789 - root_mean_squared_error: 0.1119
Epoch 2149/10000
4/4 [==============================] - 0s 46ms/step - loss: 0.1766 - mae: 0.0826 - root_mean_squared_error: 0.1183
Epoch 2150/10000
4/4 [==============================] - 0s 39ms/step - loss: 0.1738 - mae: 0.0778 - root_mean_squared_error: 0.1069
Epoch 2151/10000
4/4 [==============================] - 0s 39ms/step - loss: 0.1731 - mae: 0.0757 - root_mean_squared_error: 0.1028


Reminder: RMSE should be multiplied by 4.5

```

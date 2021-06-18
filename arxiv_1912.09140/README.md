An implementation of
[Meta Decision Trees for Explainable Recommendation Systems](https://arxiv.org/abs/1912.09140)


The implementation follows the paper pretty closely. The main difference are:
* Outputs are scaled to the interval [0, 1] (one must scale RMSE back to get proper result)
* Added a penalty for non-even split at internal nodes. It should promote splits that are more 50-50


Simple demo:

```
./demo.py
```

It should produce something along this output:

```
Epoch 1/2000
1/1 [==============================] - 12s 12s/step - loss: 0.2419 - mae: 0.2604 - root_mean_squared_error: 0.3032
Epoch 2/2000
1/1 [==============================] - 0s 275ms/step - loss: 0.2041 - mae: 0.1867 - root_mean_squared_error: 0.2326
Epoch 3/2000
1/1 [==============================] - 0s 289ms/step - loss: 0.2164 - mae: 0.1913 - root_mean_squared_error: 0.2576
Epoch 4/2000
1/1 [==============================] - 0s 279ms/step - loss: 0.2049 - mae: 0.1741 - root_mean_squared_error: 0.2344
Epoch 5/2000
1/1 [==============================] - 0s 279ms/step - loss: 0.1992 - mae: 0.1717 - root_mean_squared_error: 0.2219
Epoch 6/2000
1/1 [==============================] - 0s 265ms/step - loss: 0.2011 - mae: 0.1844 - root_mean_squared_error: 0.2261
Epoch 7/2000
1/1 [==============================] - 0s 283ms/step - loss: 0.1987 - mae: 0.1797 - root_mean_squared_error: 0.2207
Epoch 8/2000
1/1 [==============================] - 0s 268ms/step - loss: 0.1994 - mae: 0.1786 - root_mean_squared_error: 0.2223
Epoch 9/2000
1/1 [==============================] - 0s 287ms/step - loss: 0.1936 - mae: 0.1642 - root_mean_squared_error: 0.2089
 .
 .
 .
Epoch 486/2000
1/1 [==============================] - 0s 286ms/step - loss: 0.1659 - mae: 0.0797 - root_mean_squared_error: 0.1187
Epoch 487/2000
1/1 [==============================] - 0s 325ms/step - loss: 0.1653 - mae: 0.0779 - root_mean_squared_error: 0.1163
Epoch 488/2000
1/1 [==============================] - 0s 282ms/step - loss: 0.1656 - mae: 0.0800 - root_mean_squared_error: 0.1176
Epoch 489/2000
1/1 [==============================] - 0s 286ms/step - loss: 0.1644 - mae: 0.0757 - root_mean_squared_error: 0.1119
Epoch 490/2000
1/1 [==============================] - 0s 292ms/step - loss: 0.1648 - mae: 0.0733 - root_mean_squared_error: 0.1138
Epoch 491/2000
1/1 [==============================] - 0s 298ms/step - loss: 0.1635 - mae: 0.0686 - root_mean_squared_error: 0.1076
Epoch 492/2000
1/1 [==============================] - 0s 293ms/step - loss: 0.1635 - mae: 0.0690 - root_mean_squared_error: 0.1076
Epoch 493/2000
1/1 [==============================] - 0s 293ms/step - loss: 0.1644 - mae: 0.0708 - root_mean_squared_error: 0.1115
Epoch 494/2000
1/1 [==============================] - 0s 310ms/step - loss: 0.1628 - mae: 0.0678 - root_mean_squared_error: 0.1048


Reminder: RMSE should be multiplied by 4.5

```

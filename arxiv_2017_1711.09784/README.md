An implementation of
[Distilling a Neural Network Into a Soft Decision Tree](https://arxiv.org/abs/1711.09784)

**Caveats**:
It does not implement *complications* like *temperature*, *probabilities penalty*, or *windowing*
since the results are **already better than those reported in the paper**. Increasing tree's depth
would make them even better, e.g., one can easily get validation accuracy of 97% for `depth > 10`.

Another notable difference is how one measures trees' depth. This implementation's *n* corresponds
to *n-1* in the papers. Namely, a tree with a root and two leaves has depth 1 in the paper, whereas
here we considere it to be 2.

**Note**: For some reason, `tf-eigen` is several times faster then `tf-mkl` on this problem.


Simple demo:

```
./demo.py --depth=11
```

It should produce something along this output:

```
Epoch 1/1000
59/59 [==============================] - 185s 383ms/step - loss: 2.0583 - acc: 0.7622 - val_loss: 1.8978 - val_acc: 0.8335
Epoch 2/1000
59/59 [==============================] - 9s 157ms/step - loss: 1.8117 - acc: 0.8449 - val_loss: 1.7197 - val_acc: 0.8745
Epoch 3/1000
59/59 [==============================] - 9s 152ms/step - loss: 1.6606 - acc: 0.8819 - val_loss: 1.5926 - val_acc: 0.9032
Epoch 4/1000
59/59 [==============================] - 9s 157ms/step - loss: 1.5470 - acc: 0.9041 - val_loss: 1.4896 - val_acc: 0.9174
Epoch 5/1000
59/59 [==============================] - 9s 157ms/step - loss: 1.4491 - acc: 0.9180 - val_loss: 1.3971 - val_acc: 0.9264
Epoch 6/1000
59/59 [==============================] - 9s 153ms/step - loss: 1.3597 - acc: 0.9265 - val_loss: 1.3117 - val_acc: 0.9318
Epoch 7/1000
59/59 [==============================] - 10s 165ms/step - loss: 1.2758 - acc: 0.9336 - val_loss: 1.2314 - val_acc: 0.9356
Epoch 8/1000
59/59 [==============================] - 9s 153ms/step - loss: 1.1962 - acc: 0.9385 - val_loss: 1.1540 - val_acc: 0.9404
Epoch 9/1000
59/59 [==============================] - 11s 191ms/step - loss: 1.1200 - acc: 0.9426 - val_loss: 1.0812 - val_acc: 0.9439
Epoch 10/1000
59/59 [==============================] - 9s 159ms/step - loss: 1.0481 - acc: 0.9458 - val_loss: 1.0122 - val_acc: 0.9468
Epoch 11/1000
59/59 [==============================] - 10s 162ms/step - loss: 0.9800 - acc: 0.9487 - val_loss: 0.9467 - val_acc: 0.9484
Epoch 12/1000
59/59 [==============================] - 10s 165ms/step - loss: 0.9153 - acc: 0.9506 - val_loss: 0.8851 - val_acc: 0.9496
Epoch 13/1000
59/59 [==============================] - 9s 159ms/step - loss: 0.8540 - acc: 0.9527 - val_loss: 0.8263 - val_acc: 0.9513
Epoch 14/1000
59/59 [==============================] - 9s 157ms/step - loss: 0.7961 - acc: 0.9545 - val_loss: 0.7711 - val_acc: 0.9532
Epoch 15/1000
59/59 [==============================] - 9s 154ms/step - loss: 0.7414 - acc: 0.9564 - val_loss: 0.7192 - val_acc: 0.9542
Epoch 16/1000
59/59 [==============================] - 9s 151ms/step - loss: 0.6898 - acc: 0.9579 - val_loss: 0.6702 - val_acc: 0.9556
Epoch 17/1000
59/59 [==============================] - 9s 150ms/step - loss: 0.6413 - acc: 0.9596 - val_loss: 0.6248 - val_acc: 0.9564
Epoch 18/1000
59/59 [==============================] - 9s 149ms/step - loss: 0.5959 - acc: 0.9611 - val_loss: 0.5820 - val_acc: 0.9573
Epoch 19/1000
59/59 [==============================] - 9s 153ms/step - loss: 0.5535 - acc: 0.9626 - val_loss: 0.5429 - val_acc: 0.9584
Epoch 20/1000
59/59 [==============================] - 9s 150ms/step - loss: 0.5139 - acc: 0.9643 - val_loss: 0.5057 - val_acc: 0.9593
Epoch 21/1000
59/59 [==============================] - 9s 152ms/step - loss: 0.4770 - acc: 0.9652 - val_loss: 0.4718 - val_acc: 0.9598
Epoch 22/1000
59/59 [==============================] - 9s 151ms/step - loss: 0.4428 - acc: 0.9663 - val_loss: 0.4402 - val_acc: 0.9601
Epoch 23/1000
59/59 [==============================] - 9s 149ms/step - loss: 0.4111 - acc: 0.9672 - val_loss: 0.4110 - val_acc: 0.9611
Epoch 24/1000
59/59 [==============================] - 9s 150ms/step - loss: 0.3817 - acc: 0.9685 - val_loss: 0.3848 - val_acc: 0.9622
Epoch 25/1000
59/59 [==============================] - 9s 152ms/step - loss: 0.3546 - acc: 0.9693 - val_loss: 0.3600 - val_acc: 0.9624
Epoch 26/1000
59/59 [==============================] - 9s 152ms/step - loss: 0.3296 - acc: 0.9704 - val_loss: 0.3374 - val_acc: 0.9616
Epoch 27/1000
59/59 [==============================] - 9s 148ms/step - loss: 0.3065 - acc: 0.9712 - val_loss: 0.3171 - val_acc: 0.9635
Epoch 28/1000
59/59 [==============================] - 9s 149ms/step - loss: 0.2853 - acc: 0.9721 - val_loss: 0.2984 - val_acc: 0.9639
Epoch 29/1000
59/59 [==============================] - 9s 148ms/step - loss: 0.2658 - acc: 0.9729 - val_loss: 0.2811 - val_acc: 0.9638
Epoch 30/1000
59/59 [==============================] - 9s 148ms/step - loss: 0.2479 - acc: 0.9741 - val_loss: 0.2653 - val_acc: 0.9630
Epoch 31/1000
59/59 [==============================] - 9s 151ms/step - loss: 0.2315 - acc: 0.9750 - val_loss: 0.2514 - val_acc: 0.9638
Epoch 32/1000
59/59 [==============================] - 9s 151ms/step - loss: 0.2163 - acc: 0.9755 - val_loss: 0.2388 - val_acc: 0.9641
Epoch 33/1000
59/59 [==============================] - 9s 151ms/step - loss: 0.2025 - acc: 0.9764 - val_loss: 0.2268 - val_acc: 0.9637
Epoch 34/1000
59/59 [==============================] - 9s 149ms/step - loss: 0.1899 - acc: 0.9772 - val_loss: 0.2165 - val_acc: 0.9644
Epoch 35/1000
59/59 [==============================] - 9s 148ms/step - loss: 0.1781 - acc: 0.9777 - val_loss: 0.2071 - val_acc: 0.9646
Epoch 36/1000
59/59 [==============================] - 9s 148ms/step - loss: 0.1674 - acc: 0.9783 - val_loss: 0.1986 - val_acc: 0.9645
Epoch 37/1000
59/59 [==============================] - 9s 148ms/step - loss: 0.1577 - acc: 0.9793 - val_loss: 0.1905 - val_acc: 0.9655
Epoch 38/1000
59/59 [==============================] - 9s 148ms/step - loss: 0.1486 - acc: 0.9796 - val_loss: 0.1837 - val_acc: 0.9656
Epoch 39/1000
59/59 [==============================] - 9s 148ms/step - loss: 0.1404 - acc: 0.9801 - val_loss: 0.1771 - val_acc: 0.9657
Epoch 40/1000
59/59 [==============================] - 9s 149ms/step - loss: 0.1327 - acc: 0.9810 - val_loss: 0.1717 - val_acc: 0.9661
Epoch 41/1000
59/59 [==============================] - 9s 148ms/step - loss: 0.1256 - acc: 0.9815 - val_loss: 0.1666 - val_acc: 0.9665
Epoch 42/1000
59/59 [==============================] - 9s 150ms/step - loss: 0.1191 - acc: 0.9820 - val_loss: 0.1621 - val_acc: 0.9657
Epoch 43/1000
59/59 [==============================] - 9s 148ms/step - loss: 0.1131 - acc: 0.9825 - val_loss: 0.1577 - val_acc: 0.9662
Epoch 44/1000
59/59 [==============================] - 9s 149ms/step - loss: 0.1076 - acc: 0.9830 - val_loss: 0.1537 - val_acc: 0.9668
Epoch 45/1000
59/59 [==============================] - 9s 148ms/step - loss: 0.1024 - acc: 0.9833 - val_loss: 0.1503 - val_acc: 0.9670
Epoch 46/1000
59/59 [==============================] - 9s 149ms/step - loss: 0.0976 - acc: 0.9841 - val_loss: 0.1471 - val_acc: 0.9677
Epoch 47/1000
59/59 [==============================] - 9s 148ms/step - loss: 0.0932 - acc: 0.9844 - val_loss: 0.1443 - val_acc: 0.9668
Epoch 48/1000
59/59 [==============================] - 9s 148ms/step - loss: 0.0890 - acc: 0.9851 - val_loss: 0.1423 - val_acc: 0.9672
Epoch 49/1000
59/59 [==============================] - 9s 150ms/step - loss: 0.0852 - acc: 0.9854 - val_loss: 0.1401 - val_acc: 0.9669
Epoch 50/1000
59/59 [==============================] - 9s 148ms/step - loss: 0.0816 - acc: 0.9858 - val_loss: 0.1385 - val_acc: 0.9674
Epoch 51/1000
59/59 [==============================] - 9s 150ms/step - loss: 0.0782 - acc: 0.9862 - val_loss: 0.1365 - val_acc: 0.9673
Epoch 52/1000
59/59 [==============================] - 9s 148ms/step - loss: 0.0750 - acc: 0.9866 - val_loss: 0.1346 - val_acc: 0.9674
Epoch 53/1000
59/59 [==============================] - 9s 148ms/step - loss: 0.0722 - acc: 0.9870 - val_loss: 0.1330 - val_acc: 0.9681
Epoch 54/1000
59/59 [==============================] - 9s 147ms/step - loss: 0.0694 - acc: 0.9876 - val_loss: 0.1312 - val_acc: 0.9681
Epoch 55/1000
59/59 [==============================] - 9s 149ms/step - loss: 0.0669 - acc: 0.9878 - val_loss: 0.1304 - val_acc: 0.9688
Epoch 56/1000
59/59 [==============================] - 9s 149ms/step - loss: 0.0643 - acc: 0.9884 - val_loss: 0.1301 - val_acc: 0.9681
Epoch 57/1000
59/59 [==============================] - 9s 149ms/step - loss: 0.0621 - acc: 0.9887 - val_loss: 0.1284 - val_acc: 0.9688
Epoch 58/1000
59/59 [==============================] - 9s 149ms/step - loss: 0.0599 - acc: 0.9891 - val_loss: 0.1281 - val_acc: 0.9688
Epoch 59/1000
59/59 [==============================] - 9s 150ms/step - loss: 0.0578 - acc: 0.9894 - val_loss: 0.1271 - val_acc: 0.9696
Epoch 60/1000
59/59 [==============================] - 9s 148ms/step - loss: 0.0559 - acc: 0.9897 - val_loss: 0.1268 - val_acc: 0.9692
Epoch 61/1000
59/59 [==============================] - 9s 151ms/step - loss: 0.0539 - acc: 0.9901 - val_loss: 0.1257 - val_acc: 0.9700
Epoch 62/1000
59/59 [==============================] - 9s 149ms/step - loss: 0.0522 - acc: 0.9903 - val_loss: 0.1258 - val_acc: 0.9695
Epoch 63/1000
59/59 [==============================] - 9s 148ms/step - loss: 0.0506 - acc: 0.9906 - val_loss: 0.1254 - val_acc: 0.9695
Epoch 64/1000
59/59 [==============================] - 9s 148ms/step - loss: 0.0490 - acc: 0.9908 - val_loss: 0.1250 - val_acc: 0.9702
Epoch 65/1000
59/59 [==============================] - 9s 149ms/step - loss: 0.0474 - acc: 0.9911 - val_loss: 0.1249 - val_acc: 0.9701
Epoch 66/1000
59/59 [==============================] - 9s 148ms/step - loss: 0.0460 - acc: 0.9913 - val_loss: 0.1254 - val_acc: 0.9691
Epoch 67/1000
59/59 [==============================] - 9s 148ms/step - loss: 0.0446 - acc: 0.9917 - val_loss: 0.1242 - val_acc: 0.9701
Epoch 68/1000
59/59 [==============================] - 9s 157ms/step - loss: 0.0433 - acc: 0.9918 - val_loss: 0.1239 - val_acc: 0.9700
Epoch 69/1000
59/59 [==============================] - 10s 163ms/step - loss: 0.0421 - acc: 0.9922 - val_loss: 0.1243 - val_acc: 0.9702
Epoch 70/1000
59/59 [==============================] - 9s 156ms/step - loss: 0.0408 - acc: 0.9925 - val_loss: 0.1246 - val_acc: 0.9699
Epoch 71/1000
59/59 [==============================] - 9s 153ms/step - loss: 0.0397 - acc: 0.9926 - val_loss: 0.1247 - val_acc: 0.9702
Epoch 72/1000
59/59 [==============================] - 9s 156ms/step - loss: 0.0387 - acc: 0.9928 - val_loss: 0.1245 - val_acc: 0.9700
Epoch 73/1000
59/59 [==============================] - 9s 157ms/step - loss: 0.0375 - acc: 0.9931 - val_loss: 0.1245 - val_acc: 0.9700
Epoch 74/1000
59/59 [==============================] - 9s 150ms/step - loss: 0.0364 - acc: 0.9933 - val_loss: 0.1244 - val_acc: 0.9704
Epoch 75/1000
59/59 [==============================] - 9s 149ms/step - loss: 0.0356 - acc: 0.9935 - val_loss: 0.1245 - val_acc: 0.9700
Epoch 76/1000
59/59 [==============================] - 9s 151ms/step - loss: 0.0347 - acc: 0.9936 - val_loss: 0.1246 - val_acc: 0.9697
Epoch 77/1000
59/59 [==============================] - 9s 147ms/step - loss: 0.0337 - acc: 0.9939 - val_loss: 0.1252 - val_acc: 0.9698
Epoch 78/1000
59/59 [==============================] - 9s 153ms/step - loss: 0.0328 - acc: 0.9941 - val_loss: 0.1257 - val_acc: 0.9692
Epoch 79/1000
59/59 [==============================] - 9s 156ms/step - loss: 0.0320 - acc: 0.9943 - val_loss: 0.1253 - val_acc: 0.9699
Epoch 80/1000
59/59 [==============================] - 9s 156ms/step - loss: 0.0312 - acc: 0.9943 - val_loss: 0.1247 - val_acc: 0.9700
Epoch 81/1000
59/59 [==============================] - 9s 159ms/step - loss: 0.0304 - acc: 0.9944 - val_loss: 0.1251 - val_acc: 0.9699
Epoch 82/1000
59/59 [==============================] - 9s 156ms/step - loss: 0.0297 - acc: 0.9947 - val_loss: 0.1252 - val_acc: 0.9693
Epoch 83/1000
59/59 [==============================] - 9s 158ms/step - loss: 0.0289 - acc: 0.9949 - val_loss: 0.1254 - val_acc: 0.9695
Epoch 84/1000
59/59 [==============================] - 9s 157ms/step - loss: 0.0282 - acc: 0.9951 - val_loss: 0.1261 - val_acc: 0.9700


```

#-*-coding:utf-8-*-
import sys
sys.path.append("c:/users/hunho/OneDrive/문서/GitHub/supersuper/firstgit/Machine")
import numpy as np
from mnist import load_mnist

def CEE(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)

    batch_size = y_shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

(x_train,t_train),(x_test,t_test)=\
    load_mnist(normalize=True,one_hot_label=True)
print(x_train.shape)
print(t_train.shape)

print(np.random.choice(60000, 10))

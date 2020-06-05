# 需求
1. [python3](https://www.python.org/download/releases/3.0/)
2. [CuPy](https://cupy.chainer.org/) or [NumPy](https://numpy.org/)
    1. 使用[CuPy](https://cupy.chainer.org/)需先安裝[CUDA](https://developer.nvidia.com/cuda-downloads)及[cuDNN](https://developer.nvidia.com/cudnn)
    2. 使用[NumPy](https://numpy.org/)須將`Base.py`內
        ```
        import cupy as np
        # import numpy as np
        ```
        改為
        ```
        # import cupy as np
        import numpy as np
        ```


# 使用方式
> \> git sublmodule add [git@github.com:ss32324/dl_lib.git](https://github.com/ss32324/dl_lib) dl_lib

## 可使用
- [層](https://github.com/ss32324/dl_lib/blob/master/Layer.py)
    - [連接層](https://github.com/ss32324/dl_lib/blob/master/layer/FullyConnection.py)
        - Dense
        - Conv
    - [激活層](https://github.com/ss32324/dl_lib/blob/master/layer/Activation.py)
        - Sigmoid
        - Softmax
        - ReLU
    - [正規層](https://github.com/ss32324/dl_lib/blob/master/layer/Normalization.py)
        - BatchNorm
        - LayerNorm
    - [池化層](https://github.com/ss32324/dl_lib/blob/master/layer/Pool.py)
        - MaxPooling
    - 平坦層
        - Flatten
    - ？？層
        - Dropout
- [優化器](https://github.com/ss32324/dl_lib/blob/master/Optimizer.py)
    - SGD
    - MBGD
    - Adam
- [損失函數](https://github.com/ss32324/dl_lib/blob/master/LossFunc.py)
    - MSE
    - CategoricalCrossEntropy
    - BinaryCrossEntropy
- [標籤](https://github.com/ss32324/dl_lib/blob/master/PreData.py)
    - OneHot

## 引入
```python
from dl_lib import *
```

## 建立模組
```python
model = Model(layers = (
    Conv(k_no=8, k_size=3, is_add_padding=True),
    ReLU(),
    Conv(k_no=8, k_size=3, is_add_padding=True),
    ReLU(),
    MaxPooling(2),

    Flatten(),

    Dense(128),
    BatchNorm(),
    ReLU(),
    Dropout(0.5),

    Dense(10),
    Softmax(),
))

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    lossfunc=CategoricalCrossEntropy()
)
```

## 訓練
```python
onehot = OneHot(np.arange(0,10))

train_y, test_y = onehot.encoding(train_y), onehot.encoding(test_y)

# history return like {'los' = [...], 'val_los' = [...], 'acc' = [...], 'val_acc' = [...]}
history = model.fit(train_x, train_y, test_x, test_y, epoch, batch_size, clean_opt)
```


## 評估mode 及 取得錯誤的index
```python
accuracy, loss = model.evaluate(test_x, test_y)
indexes = model.get_diff(test_x, test_y)
```

## 儲存model
```python
# Model.save(model, path)
model.save(path)
```

## 讀取model
```python
import_model = Model.load(path)
```






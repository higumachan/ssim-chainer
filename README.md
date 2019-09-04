# ssim-chainer

## Differentiable structural similarity index. reimplemantation from [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim) 

## Installation

```sh
pip install git+https://github.com/higumachan/ssim-chainer.git
```

## example

### maximize ssim

```python3
import numpy as np
import cv2
import chainer
import chainer.links as L
import chainer.functions as F

from chainer import Variable
from chainer.optimizers import Adam

from ssim.functions import ssim_loss
import matplotlib.pyplot as plt


def loss(y, t):
    return -ssim_loss(y, t, 11, 1)


if __name__ == '__main__':
    os.system("ls assets/einstein.png")
    img1 = cv2.imread("assets/einstein.png")

    img1 = img1.astype(np.float32).transpose(2, 0, 1) / 255.0
    img1 = np.expand_dims(img1, 0)
    img1 = Variable(img1)

    img2 = L.Parameter(np.random.rand(*img1.shape).astype(np.float32))

    ssim_value = ssim_loss(img1, img2(), 11, 11)
    print("Initial ssim:", ssim_value)

    optimizer = Adam(0.1)
    optimizer.setup(img2)

    step = 1
    while ssim_value.data < 0.95:
        optimizer.update(loss, img1, img2())
        ssim_value = -loss(img1, img2())

        ssim_value_s = "ssim: {}".format(ssim_value.array)
        print("ssim:", ssim_value)

        im = (img2.W.array[0].transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)
        plt.imshow(im)
        plt.text(0, -5, ssim_value_s)
        plt.show()

        step += 1

```

# Reference

[https://github.com/Po-Hsun-Su/pytorch-ssim/](https://github.com/Po-Hsun-Su/pytorch-ssim/)
[https://ece.uwaterloo.ca/~z70wang/research/ssim/](https://ece.uwaterloo.ca/~z70wang/research/ssim/)

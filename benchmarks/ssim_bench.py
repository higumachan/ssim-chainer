import numpy
from benchmarker import Benchmarker
from chainer import Variable

from ssim.functions import ssim_depthwise_convolution, ssim_im2col

shape = (1, 3, 256, 256)

with Benchmarker(cycle=3, extra=1, filter="tag!=gpu") as bench:
    @bench('ssim depthwise convolution')
    def _(bm):
        y = Variable(numpy.random.rand(*shape))
        t = Variable(numpy.random.rand(*shape))
        with bm:
            ssim_depthwise_convolution(y, t, 11, 1)


    @bench('ssim im2col')
    def _(bm):
        y = Variable(numpy.random.rand(*shape))
        t = Variable(numpy.random.rand(*shape))
        with bm:
            ssim_im2col(y, t, 11, 1)

    @bench('gpu ssim depthwise convolution', tag=("gpu",))
    def _(bm):
        y = Variable(numpy.random.rand(*shape))
        t = Variable(numpy.random.rand(*shape))
        y.to_device(0)
        t.to_device(0)
        with bm:
            ssim_depthwise_convolution(y, t, 11, 1)


    @bench('gpu ssim im2col', tag=("gpu",))
    def _(bm):
        y = Variable(numpy.random.rand(*shape))
        t = Variable(numpy.random.rand(*shape))
        y.to_device(0)
        t.to_device(0)
        with bm:
            ssim_im2col(y, t, 11, 1)

if __name__ == '__main__':
    pass

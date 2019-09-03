import chainer.functions as F


def _create_window(window_size, channel, xp):
    return xp.ones((1, channel, window_size, window_size)) / (window_size ** 2)


def ssim_loss(y, t, window_size, stride):
    n, c, w, h = y.shape
    window = _create_window(window_size, c, y.xp)
    mu_y = F.depthwise_convolution_2d(y, window, stride=stride)
    mu_t = F.depthwise_convolution_2d(t, window, stride=stride)
    mu_y_sq = F.depthwise_convolution_2d(y * y, window, stride=stride)
    mu_t_sq = F.depthwise_convolution_2d(t * t, window, stride=stride)
    mu_ty = F.depthwise_convolution_2d(y * t, window, stride=stride)
    muy_mut = (mu_y * mu_t)

    sq_mu_y = mu_y ** 2
    sq_mu_t = mu_t ** 2

    sigma_y_sq = mu_y_sq - sq_mu_y
    sigma_t_sq = mu_t_sq - sq_mu_t
    sigma_yt = mu_ty - muy_mut

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * muy_mut + c1) * (2 * sigma_yt + c2)) / ((sq_mu_y + sq_mu_t + c1) * (sigma_y_sq + sigma_t_sq + c2))
    print(ssim_map)

    return F.mean(ssim_map)


if __name__ == '__main__':
    import numpy as np
    import chainer
    y = chainer.Variable(np.ones([1, 3, 3, 3]))
    t = chainer.Variable(np.ones([1, 3, 3, 3]))
    assert ssim_loss(y, t, 3, 3).data == 1.0

    y = chainer.Variable(np.random.random((1, 1, 3, 3)))
    t = chainer.Variable(np.ones([1, 1, 3, 3]))

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    print(y.data, t.data)
    cov = (t.data - t.data.mean()) * (y.data - y.data.mean())
    expect = ((2 * t.data.mean() * y.data.mean() + c1) * (2 * cov + c2)) / ((y.data.mean() ** 2 + t.data.mean() ** 2 + c1) * (y.data.var() + t.data.var() + c2))

    loss = ssim_loss(y, t, 3, 3).data
    print(expect.mean(), loss)
    assert expect.mean() == loss

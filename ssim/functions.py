import chainer.functions as F


def create_window(window_size, channel, xp):
    return xp.ones((1, channel, window_size, window_size))


def ssim_loss(y, t, window_size, stride):
    n, c, w, h = y.shape
    mu_y = F.depthwise_convolution_2d(y, create_window(window_size, c, y.xp), stride=stride)
    mu_t = F.depthwise_convolution_2d(t, create_window(window_size, c, y.xp), stride=stride)
    mu_y_sq = F.depthwise_convolution_2d(y * y, create_window(window_size, c, y.xp), stride=stride)
    mu_t_sq = F.depthwise_convolution_2d(t * t, create_window(window_size, c, y.xp), stride=stride)
    mu_ty = F.depthwise_convolution_2d(y * t, create_window(window_size, c, y.xp), stride=stride)
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
    print(ssim_loss(y, t, 3, 3))

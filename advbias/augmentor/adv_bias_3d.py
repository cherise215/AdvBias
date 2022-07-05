import torch
import torch.nn.functional as F
import numpy as np

from advbias.augmentor.adv_bias import AdvBias


"""
    Create a 3d bspline kernel matrix
"""


def bspline_kernel_3d(sigma=[1, 1, 1], order=2, asTensor=False, dtype=torch.float32, device='gpu'):
    kernel_ones = torch.ones(1, 1, *sigma)
    kernel = kernel_ones
    padding = np.array(sigma) - 1

    for i in range(1, order + 1):
        # change 2d to 3d
        kernel = F.conv3d(kernel, kernel_ones, padding=(
            padding).tolist())/(sigma[0]*sigma[1]*sigma[2])
    if asTensor:
        return kernel[0, 0, ...].to(dtype=dtype, device=device)
    else:
        return kernel[0, 0, ...].numpy()


class AdvBias3D(AdvBias):
    """
     Adv Bias field 
    """

    def __init__(self,
                 config_dict={
                     'epsilon': 0.3,  # bias field magnitude
                     # spacings between two points along z, y and x direction.
                     'control_point_spacing': [64, 64, 64],
                     # we downsample images to reduce computational costs, especially with large spacing along images. To get true spacings, one should multiply control point spacing with downscale.
                     'downscale': 2,
                     'data_size': [2, 1, 128, 128, 128],
                     'interpolation_order': 3,
                     'init_mode': 'random',
                     'space': 'log'},
                 power_iteration=False,
                 use_gpu=True, debug=False):
        super(AdvBias, self).__init__(
            config_dict=config_dict, use_gpu=use_gpu, debug=debug)
        self.power_iteration = power_iteration

    def compute_smoothed_bias(self, cpoint=None, interpolation_kernel=None, padding=None, stride=None):
        '''
        generate bias field given the cpoints N*1*k*l
        :return: bias field bs*1*H*W
        '''
        if interpolation_kernel is None:
            interpolation_kernel = self.interp_kernel
        if padding is None:
            padding = self._padding
        if stride is None:
            stride = self._stride
        if cpoint is None:
            cpoint = self.param
        bias_field = F.conv_transpose3d(cpoint, interpolation_kernel,
                                        padding=padding, stride=stride, groups=1)
        # crop bias
        bias_field_tmp = bias_field[:, :,
                                    stride[0] + self._crop_start[0]:-stride[0] - self._crop_end[0],
                                    stride[1] + self._crop_start[1]:-stride[1] - self._crop_end[1],
                                    stride[2] + self._crop_start[2]:-stride[2] - self._crop_end[2],
                                    ]

        # recover bias field to original image resolution for efficiency.
        if self.debug:
            print('after bspline intep, size:', bias_field_tmp.size())
        scale_factor_d = self._image_size[0] / bias_field_tmp.size(2)
        scale_factor_h = self._image_size[1] / bias_field_tmp.size(3)
        scale_factor_w = self._image_size[2] / bias_field_tmp.size(4)

        if scale_factor_h > 1 or scale_factor_w > 1:
            upsampler = torch.nn.Upsample(scale_factor=(scale_factor_d, scale_factor_h, scale_factor_w), mode='trilinear',
                                          align_corners=False)
            diff_bias = upsampler(bias_field_tmp)
            print('recover resolution, size of bias field:', diff_bias.size())

        else:
            diff_bias = bias_field_tmp

        if self.use_log:
            bias_field = torch.exp(diff_bias)
        else:
            bias_field=1+diff_bias
        return bias_field

    def get_bspline_kernel(self, spacing, order=3):
        '''

        :param order init: bspline order, default to 3
        :param spacing tuple of int: spacing between control points along h and w.
        :return:  kernel matrix
        '''
        self._kernel = bspline_kernel_3d(
            spacing, order=order, asTensor=True, dtype=self._dtype, device=self._device)
        self._padding = (np.array(self._kernel.size()) - 1) / 2
        self._padding = self._padding.astype(dtype=int).tolist()
        self._kernel.unsqueeze_(0).unsqueeze_(0)
        self._kernel = self._kernel.to(dtype=self._dtype, device=self._device)
        return self._kernel

    def get_name(self):
        return 'bias'

    def is_geometric(self):
        return 0


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    images = 128*torch.randn(2, 1, 128, 128, 128).cuda()
    images[:, :, 10:120, 10:120, 10:120] =256
    images =images.clone()

    images = images.float()
    images.requires_grad = False
    print('input:', images.size())
    augmentor = AdvBias3D(
        config_dict={'epsilon': 0.3,
                     'control_point_spacing': [64, 64, 64],
                     'downscale': 4,  # increase the downscale factor to save interpolation time
                     'data_size': [2, 1, 128, 128, 128],
                     'interpolation_order': 3,
                     'init_mode': 'random',
                     'space': 'log'},
        power_iteration=False,
        debug=True, use_gpu=True)

    # perform random bias field
    augmentor.init_parameters()
    transformed = augmentor.forward(images)
    error = transformed-images
    print('sum error', torch.sum(error))

    plt.subplot(231)
    plt.imshow(images.detach().cpu().numpy()[0, 0, 0])
    plt.title("Input slice: 0 ")

    plt.subplot(232)
    plt.imshow(transformed.detach().cpu().numpy()[0, 0, 0])
    plt.title("Augmented: 0")

    plt.subplot(233)
    plt.imshow((augmentor.bias_field.detach()).detach().cpu().numpy()[0, 0, 0])
    plt.title("Bias Field: 0")

    plt.subplot(234)
    plt.imshow(images.detach().cpu().numpy()[0, 0, 28])
    plt.title("Input slice: 28")

    plt.subplot(235)
    plt.imshow(transformed.detach().cpu().numpy()[0, 0, 28])
    plt.title("Augmented: 28")

    plt.subplot(236)
    plt.imshow(augmentor.bias_field.detach().cpu().numpy()[0, 0, 28])
    plt.title("Bias field: 28")

    plt.savefig('./result/test_bias_3D.png')

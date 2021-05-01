# -*- coding: utf-8 -*-
"""
Created on %(30.04.21)s

@author: %(Igor Ratajczyk)s
"""
from typing import Optional, Union
from cv2 import dft, idft, DFT_SCALE, DFT_COMPLEX_OUTPUT, DFT_COMPLEX_INPUT, BORDER_CONSTANT, copyMakeBorder, merge, \
    morphologyEx, MORPH_TOPHAT, magnitude, filter2D
from matplotlib.pyplot import subplots, show
from matplotlib.patches import Rectangle
from numpy import rot90, float32, real, imag, ones, uint8, mgrid, exp, max as max_, abs as abs_, ndarray
from numpy.fft import fftshift, ifftshift


def FT2DShifted(image: ndarray) -> ndarray:
    """
    Function performing 2-dimensional Fourier Transform with shift in frequency domain, required in future processing.
    :param image: numpy.ndarray: preferably monochromatic image
    :return: numpy.ndarray: 2D-Fourier Transform with proper shift
    """
    return fftshift(dft(float32(image), flags=DFT_COMPLEX_OUTPUT), [0, 1])


def fgaussian(size, sigma):
    m, n = size[0] * 2, size[1] * 2
    h, k = m // 2, n // 2
    x, y = mgrid[-h:h + 1, -k:k + 1]
    g = exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def FastPatternRecognition(image: ndarray, pattern: ndarray) -> ndarray:
    """
    Function dedicated to performing fast pattern recognition.
    :param image: numpy.ndarray: Monochromatic image with potential pattern samples inside.
    :param pattern: numpy.ndarray: Monochromatic image with the very pattern looked for, it is highly recommended to pass it without redundand borders.
    :return: numpy.ndarray: Basically processed image, suitable for future processing.
    """
    image_Fourier_domain = FT2DShifted(image)
    pattern_rotated = rot90(pattern, 2)
    pattern_padded = copyMakeBorder(pattern_rotated, 0, image.shape[0] - pattern.shape[0], 0,
                                    image.shape[1] - pattern.shape[1], BORDER_CONSTANT, value=0)
    pattern_Fourier_domain = FT2DShifted(pattern_padded)
    image_Cmplx = image_Fourier_domain[:, :, 0] + 1j * image_Fourier_domain[:, :, 1]
    pattern_Cmplx = pattern_Fourier_domain[:, :, 0] + 1j * pattern_Fourier_domain[:, :, 1]
    output_Cmplx = image_Cmplx * pattern_Cmplx
    output_shifted = ifftshift(merge([real(output_Cmplx), imag(output_Cmplx)]), [0, 1])
    output_time_domain = idft(output_shifted, flags=DFT_SCALE | DFT_COMPLEX_OUTPUT | DFT_COMPLEX_INPUT)
    output_abs = abs_(output_time_domain)
    final_result = morphologyEx(output_abs, MORPH_TOPHAT, ones((3, 1), uint8))
    return final_result


def Comparison(image: ndarray, pattern: ndarray, Recognised_pattern_img: Optional[ndarray] = None) -> None:
    """
    Function dedicated to comapre both input image and raw FPR function result.
    :param image: numpy.ndarray: Monochromatic image with potential pattern samples inside.
    :param pattern: numpy.ndarray: Monochromatic image with the very pattern looked for, it is highly recommended to pass it without redundand borders.
    :param Recognised_pattern_img: The default is None. Had FastPatterRecognition already been computed, it is recommended to pass it as third argument, otherwise leave it ot pass None.
    :return: None
    """
    patter_recognition = FastPatternRecognition(image,
                                                pattern) if Recognised_pattern_img is None else Recognised_pattern_img

    f, (ax1, ax2) = subplots(1, 2, figsize=(10, 10))
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Base img')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.imshow(magnitude(patter_recognition[:, :, 0], patter_recognition[:, :, 1]))
    ax2.set_title('Final img', cmap='gray')
    ax2.set_xticks([])
    ax2.set_yticks([])

    show()


def naive_FPR_thresholding(image, pattern, processed_image=None, threshold=-1, gauss_filter_size = -1,
                           gauss_filter_std: Union[int, float] = 20):
    """

    :param image: numpy.ndarray: Monochromatic image with potential pattern samples inside.
    :param pattern: numpy.ndarray: Monochromatic image with the very pattern looked for, it is highly recommended to pass it without redundand borders.
    :param processed_image: The default is None. Had FastPatterRecognition already been computed, it is recommended to pass it as third argument, otherwise leave it ot pass None.
    :param threshold: The default is -1.
    :param gauss_filter_size: The default is -1. Specify the size of Gaussian filter used after thresholding by passing Tuple consistion of pair of positive integers.
    :param gauss_filter_std: The default is 20. Specify the standard deviation of Gaussian filter used after thresholding.
    :return:
    """
    image_magnitude = magnitude(processed_image[:, :, 0], processed_image[:, :, 1])
    binar_result = ((image_magnitude > threshold) * 255).astype(uint8)

    processed_image = FastPatternRecognition(image, pattern) if processed_image is None else processed_image
    gauss_filter_size = pattern.shape if gauss_filter_size == -1 else gauss_filter_size
    threshold = max_(image_magnitude) if threshold == -1 else threshold

    filter_ = fgaussian(pattern.shape, gauss_filter_std)
    result_filtered = filter2D(binar_result, -1, filter_)

    f, (ax1, ax2) = subplots(1, 2, figsize=(10, 10))
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Base img')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.imshow(result_filtered, cmap='gray')
    ax2.set_title('Pattern')
    ax2.set_xticks([])
    ax2.set_yticks([])

    t = max_(result_filtered) - 1
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if result_filtered[i, j] > t:
                rect = Rectangle((j - (pattern.shape[1]), i - (pattern.shape[0])), pattern.shape[1], pattern.shape[0],
                                 linewidth=1, edgecolor='r', facecolor='none')
                ax1.add_patch(rect)
    show()

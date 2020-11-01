import numpy as np
import scipy.io.wavfile as sc_wav
from scipy import signal
from imageio import imread
import skimage.color as skimage
from scipy.ndimage.interpolation import map_coordinates

CHANGE_SAMPLES_FILE_NAME = "change_samples.wav"
CHANGE_RATE_FILE_NAME = "change_rate.wav"
NORMALIZED = 255.0
GRAY_SCALE = 1
CONVOLUTION_VECTOR = np.array([-0.5, 0, 0.5])


# ex2_helper:
def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio
    # time_steps = np.arange(spec.shape[1]) * ratio
    # time_steps = time_steps[time_steps < spec.shape[1]]

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect',
                                  order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec


def read_image(filename, representation):
    """
    This function reads an image file and converts it into a given representation.
    :param filename: the filename of an image on disk(could be grayscale or RGB)
    :param representation: representation code, either 1 or 2 defining whether the
        output should be a grayscale image(1) or an RGB image(2).
    :return: An image, represented by a matrix of type np.float64 with intensities.
    """
    image = imread(filename)

    # checks if the image is already from type float64
    if not isinstance(image, np.float64):
        image.astype(np.float64)
        image = image / NORMALIZED

    # checks if the output image should be grayscale
    if representation == GRAY_SCALE:
        image = skimage.rgb2gray(image)
    return image


def fourier_transform(signal, normalized, sign, N):
    """
    This function operates the fourier transform equation
    :param signal: an array representing the signal/ fourier signal to transform
    :param normalized: regular normalization
    :param sign: the sign of the exponent
    :return: an array with the fourier representation of the given signal or vice
        versa
    """

    x = np.arange(N)
    u = x.reshape((N, 1))
    exp = (np.exp(((sign * 2j * np.pi) / N) * x * u)) / normalized
    return np.dot(exp, signal)


def DFT(signal):
    """
    This function transform a 1D discrete signal to its fourier representation
    :param signal: an array of dtype float64 with shape (N,1)
    :return: an array with the fourier representation of the given signal
    """
    N = len(signal)
    return fourier_transform(signal, 1, -1, N)


def IDFT(fourier_signal):
    """
    This function inverses a 1D fourier signal to its original domain
    :param fourier_signal: an array of dtype complex128 with shape (N,1)
    :return: an array with the original signal of the given fourier signal
    """
    N = len(fourier_signal)
    return np.real_if_close(
        fourier_transform(fourier_signal.astype(np.complex128), N, 1, N))


def DFT2(image):
    """
    This function transform a 2D discrete signal to its fourier representation
    :param image: a grayscale image with dtype float64
    :return: a 2D array with the fourier representation of the given signal
    """
    matrix = DFT(image)
    return DFT(matrix.transpose()).transpose()


def IDFT2(fourier_image):
    """
    This function inverses a 2D fourier signal to its original domain
    :param fourier_image: a 2D array of dtype complex128
    :return: a 2D array with the original image of the given fourier image
    """
    matrix = IDFT(fourier_image)
    return IDFT(matrix.transpose()).transpose()


def change_rate(filename, ratio):
    """
    This function changes the duration of an audio file by keeping the same samples,
    but changing the sample rate written in the file header. The new audio file will
    be saved in a new file called change_rate.wav
    :param filename: a string representing the path to a WAV file
    :param ratio: a positive float64 representing the duration change
    """
    sample_rate, data = sc_wav.read(filename)
    sample_rate *= ratio
    sc_wav.write(CHANGE_RATE_FILE_NAME, int(sample_rate), data)


def resize(data, ratio):
    """
    This function changes the number of samples by a given ratio.
    :param data: a 1D array of dtype float64 or complex128 representing the original
        sample points
    :param ratio: a positive float64 representing the duration change
    :return: a 1D ndarray of the dtype of data representing the new sample points
    """
    orig_sample_amount = len(data)
    trans_data = DFT(data)
    shift_data = np.fft.fftshift(trans_data)

    # finds the index of the center after the shift
    center_index = np.where(shift_data == trans_data[0])[0][0]
    new_samples_amount = int(np.floor(orig_sample_amount / ratio))

    new_samples_amount_data = create_new_sample_arr(center_index, new_samples_amount,
                                                    orig_sample_amount, shift_data)
    new_samples_ishift = np.fft.ifftshift(new_samples_amount_data)
    return IDFT(new_samples_ishift).astype(data.dtype)


def create_new_sample_arr(center_index, new_samples_amount, orig_sample_amount,
                          shift_data):
    """
    This method changes the size of the sample array according to the new size
    calculated using a given ratio
    :param center_index: the index of the center after the shift
    :param new_samples_amount: the amount of samples in the array after the change
        according to the ratio
    :param orig_sample_amount: the original amount of samples
    :param shift_data: the data array (samples) after DFT and shift
    :return: the data(samples) array after the resize
    """
    if new_samples_amount % 2 == 0:
        even = True
    else:
        even = False
    # checks if the new sample rate array should be smaller
    if orig_sample_amount > new_samples_amount:
        if even:
            new_samples_amount_data = shift_data[center_index - (
                int(new_samples_amount / 2)):
                                                 center_index + int(
                                                     (new_samples_amount / 2))]
        else:
            new_samples_amount_data = shift_data[center_index - (
                int(np.floor(new_samples_amount / 2))):
                                                 center_index + int(np.floor(
                                                     (new_samples_amount / 2))) + 1]

    # else, it should be bigger, add zeros on the sides
    else:
        zero_size = int(np.floor((new_samples_amount - orig_sample_amount) / 2))
        zeros_array_left = [0] * zero_size
        if even:
            zeros_array_right = [0] * zero_size
        else:
            zeros_array_right = [0] * (zero_size + 1)
        new_samples_amount_data = np.array(
            zeros_array_left + list(shift_data) + zeros_array_right)
    return new_samples_amount_data


def change_samples(filename, ratio):
    """
    This function changes the duration of an audio file by reducing the numbers of
    samples using Fourier.
    :param filename: a string representing the path to a WAV file
    :param ratio: a positive float64 representing the duration change
    :return: a 1D ndarray of dtype float64 representing the new samples points
    """
    sample_rate, data = sc_wav.read(filename)
    new_data = resize(np.array(data), ratio).astype(np.float64)
    sc_wav.write(CHANGE_SAMPLES_FILE_NAME, sample_rate, new_data.astype(np.int16))
    return new_data


def resize_spectrogram(data, ratio):
    """
    This function speeds up a WAV file, without changing the pitch, using spectrogram
    scaling
    :param data: a 1D ndarray of dtype float64 representing the original samples points
    :param ratio: a positive float64 representing the rate change of the WAV file
    :return: a new sample points according to ratio
    """
    spec = stft(data)
    resize_spec = np.apply_along_axis(resize, 1, spec, ratio)
    return istft(resize_spec)


def resize_vocoder(data, ratio):
    """
    This function speedups a WAV file by phase vocoding its spectrogram
    :param data: a 1D ndarray of dtype float64 representing the original samples points
    :param ratio: a positive float64 representing the rate change of the WAV file
    :return: the given data rescaled acc
    """
    spec = stft(data)
    warped_spec = phase_vocoder(spec, ratio)
    return istft(warped_spec)


def conv_der(im):
    """
    This function computes the magnitude of image derivatives
    :param im: grayscale image of type float64
    :return: the magnitude of image derivatives
    """
    extend_CONVOLUTION_VECTOR = np.expand_dims(CONVOLUTION_VECTOR, 0)
    dx = signal.convolve2d(im, extend_CONVOLUTION_VECTOR, mode="same")
    dy = signal.convolve2d(im, extend_CONVOLUTION_VECTOR.transpose(), mode="same")
    return calculate_magnitude(dx, dy)


def calculate_magnitude(dx, dy):
    """
    This function calculates the magnitude of image given its partial derivatives
    :param dx: partial derivative of axis x
    :param dy: partial derivative of axis y
    :return: the magnitude of this derivatives
    """
    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude


def fourier_der(im):
    """
    This function computes the magnitude of image derivatives using fourier transform
    :param im: grayscale image of type float64
    :return: magnitude of image derivatives
    """
    N = len(im)  # num of rows
    dx = calculate_derive_using_fourier(N, im, False)
    M = len(im[0])  # num of columns
    dy = calculate_derive_using_fourier(M, im, True)
    return calculate_magnitude(dx, dy)


def calculate_derive_using_fourier(N, im, col):
    """
    This function calculates a given im derivative using fourier transform
    :param N: the range to multiply the frequencies
    :param im: grayscale image of type float64
    :param col: boolean parameter that determent if the dy or dx is being calculated
        (rows or columns)
    :return: im derivative
    """
    u = np.arange(-N / 2, N / 2)
    if not col:
        u = u.reshape((N, 1))
    normalized = 2j * np.pi / N
    return normalized * IDFT2(np.fft.ifftshift(u * np.fft.fftshift(DFT2(im))))

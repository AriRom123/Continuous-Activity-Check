import scipy
import numpy as np
from scipy.ndimage import uniform_filter
import cv2 as cv
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from scipy.signal import butter, filtfilt

def bilateral_filter(img, sigma_spatial, sigma_range):
    """
    Bilateral filter to a 2D numpy array.
    """
    return cv.bilateralFilter(img.astype(np.float32), -1, sigma_spatial, sigma_range)

def gaussian_filter(img, sigma):
    """
    Gaussian filter to a 2D numpy array.
    """
    return cv.GaussianBlur(img.astype(np.float32), (0, 0), sigma)

def average_filter(img, kernel_size):
    """
    Average filter to a 2D numpy array.
    """
    return cv.blur(img, (kernel_size, kernel_size))

def normalize_box_blur(img, kernel_size):
    """
    Normalize box blur to a 2D numpy array.
    """
    return cv.boxFilter(img, -1, (kernel_size, kernel_size), normalize=True)



#### Imported modules to main code

def resize(img ,N,M):
    return cv.resize(img, (N,M), interpolation = cv.INTER_AREA)

def Filter(img,fil,sigma,kernel):
    if fil == 'gaussian':
        return gaussian_filter(img, sigma=sigma)
    if fil == 'bilateral':
        return bilateral_filter(img, sigma_spatial=sigma, sigma_range=7)
    if fil == 'average':
        return average_filter(img, kernel_size=kernel)
    if fil == 'normalize':
        return normalize_box_blur(img, kernel_size=kernel)
    ## Sigma - for gaussian and bilateral
    ## kernel - for average and normalize

def pca(img,n_comp):
    pca = PCA(n_components = n_comp)
    components = pca.fit_transform(img)
    return pca.inverse_transform(components)

def decrease_frame_rate(video_array, original_fps, target_fps):
    # Calculate the frame skip factor
    skip_factor = original_fps // target_fps

    cutoff_frequency = target_fps / (original_fps)

    # Define a low-pass filter
    nyquist = 0.5 * original_fps
    normal_cutoff = cutoff_frequency / nyquist

    b, a = scipy.signal.butter(2, normal_cutoff, btype='low', analog=False)

    # Apply the filter to each pixel independently across time
    filtered_frames = np.zeros_like(video_array)
    for i in range(video_array.shape[0]):
        for j in range(video_array.shape[1]):
            filtered_frames[i, j, :] = scipy.signal.filtfilt(b, a, video_array[i, j, :])

    # Select every skip_factor-th frame
    reduced_video_array = filtered_frames[:, :, ::skip_factor]

    return reduced_video_array

def normalize_data(data):
    flattened_data = abs(data).flatten()

    sorted_data = np.sort(flattened_data)[::-1]  # Sort the flattened data in descending order

    percentile_index = int(len(sorted_data) * 0.001)  # Determine the index corresponding to the x-th percentile

    max_value = sorted_data[percentile_index - 1]  # Retrieve the maximum value with respect to chosen value of the data

    normalized_data = data / max_value

    return normalized_data



import numpy as np
from scipy.signal import convolve2d , convolve


def horn_schunck(im1, im2, alpha, num_iter):
    def avg(u):
        avg_stencil = np.array([[0, 1/4, 0],
                                [1/4, 0, 1/4],
                                [0, 1/4, 0]])

        return convolve2d(u, avg_stencil, mode='same', boundary='symm')

    def derivative(matrix):
        sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


        dx = convolve2d(matrix, sobel_x , mode='same', boundary='symm')
        dy = convolve2d(matrix, sobel_y , mode='same', boundary='symm')

        return dx, dy

    # Initialize the flow vectors to zero
    u = np.zeros_like(im1)
    v = np.zeros_like(im1)

    # Compute derivatives of the images using central difference
    Ix , Iy = derivative(im1)
    It = im2 - im1


    old = np.stack([u, v], axis=2)


    abs_values = np.linalg.norm(im1, axis=1)  # Compute the absolute values of each vector
    mean_abs_value = np.mean(abs_values)


    for i in range(num_iter):
    #while frobenius_norm > mean_abs_value * (0.01)**2:

        # Compute the Laplacian of the flow vectors
        u_avg = avg(u)
        v_avg = avg(v)

        Ix_uavg = np.multiply(Ix, u_avg)
        Iy_vavg = np.multiply(Iy, v_avg)
        Ix_uavg__Iy_vavg = np.add(Ix_uavg,Iy_vavg)
        Ix_uavg__Iy_vavg__It = np.add(Ix_uavg__Iy_vavg,It)

        Ix_2__Iy_2 = np.add(np.square(Ix),np.square(Iy))
        alpha__Ix_2__Iy_2 = np.add(Ix_2__Iy_2, (alpha ** 2))

        A = np.divide(Ix_uavg__Iy_vavg__It , alpha__Ix_2__Iy_2)

        u = u_avg - np.multiply(Ix, A)
        v = v_avg - np.multiply(Iy, A)

        new = np.stack([u, v], axis=2)

        diff = new - old

        frobenius_norm = np.sqrt(np.sum(diff ** 2))

        old = new.copy()
    return np.stack([u, v], axis=2)



def lucas_kanade(prev_frame, next_frame, window_size):
    # Calculate spatial gradients (Sobel filters)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    I_x = convolve(prev_frame, sobel_x)
    I_y = convolve(prev_frame, sobel_y)
    I_t = next_frame - prev_frame

    half_window = window_size // 2

    u = np.zeros(prev_frame.shape)
    v = np.zeros(prev_frame.shape)

    # Iterate over the image to calculate optical flow for each pixel
    for i in range(half_window, prev_frame.shape[0] - half_window):
        for j in range(half_window, prev_frame.shape[1] - half_window):
            # Extract window around the pixel
            I_x_window = I_x[i - half_window:i + half_window + 1, j - half_window:j + half_window + 1].flatten()
            I_y_window = I_y[i - half_window:i + half_window + 1, j - half_window:j + half_window + 1].flatten()
            I_t_window = I_t[i - half_window:i + half_window + 1, j - half_window:j + half_window + 1].flatten()

            A = np.vstack((I_x_window, I_y_window)).T
            b = -I_t_window

            # Solve the linear system
            if np.linalg.matrix_rank(np.dot(A.T, A)) >= 2:
                flow = np.dot(np.linalg.pinv(A), b)
                u[i, j] = flow[0]
                v[i, j] = flow[1]

    return np.stack([u, v], axis=2)






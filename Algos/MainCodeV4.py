import math
import cv2
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
import scipy.signal

from Algos.Create_Patterns import create_patterns, create_gaussians
from Algos.Data_Processing import Filter, pca, resize , decrease_frame_rate , normalize_data
from Algos.Display import plot_quiver, trajectory
from Algos.Horn_Schunck import horn_schunck
from Algos.longest_path import nodes_connections

from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm

plt.rcParams['animation.ffmpeg_path'] = '/opt/homebrew/bin/ffmpeg'

#### Algorithm parameters###

alpha = 1.6         ## Optic Flow Horn-Schunck alpha parameter
epsilon = 0.2      ## Saddle point epsilon threshold
iterations = 150    ## Number of iterations
n = 4               ## neighborhood size (2n+1)x(2n+1) around each pixel
theta = 10         ## Theta threshold for uniform continuity







def data(type):
    if type == '1 gaussian':
        dff1 , params = create_gaussians(N=50, M=50, frames=50 , num_gaus=1, x0=20, y0=25, sd0=5, t0_0=20, sdT0=5)
        title = fr'1 Gaussian $\sigma$={params[6]} , $\sigma_T$={params[8]}'

        return dff1 , title

    if type == '2 gaussian':
        dff1 , params = create_gaussians(N=50, M=50, frames=50, num_gaus=2, x0=20, y0=20, sd0=5, t0_0=20, sdT0=5 \
                                                                , x1=20 + 7, y1=20+7,sd1=5, t0_1=20 + 10 ,sdT1=5)

        title = r'2 Gaussians ' # $\Delta_X$=$\Delta_T$± 2$\sigma$ , $\sigma_x$ =5'

        return dff1 , title

    if type == 'radial':
        dff1 = create_patterns(N=80, M=80, frames=100, pattern='radial', x0=40, y0=40, sd0=4, u=0, v=1,rad_spd=0.3, rad_width=4)
        title = 'Radial Wave'

        return dff1 , title

    if type == 'plane':
        dff1 = create_patterns(N=80, M=80, frames=100, pattern='plane', x0=-25, y0=-25, sd0=8, u=0, v=1,rad_spd=0.3, rad_width=3)
        title = 'plane'

        return dff1 , title

    if type == 'cont':
        dff1 = create_patterns(N=50, M=50, frames=100, pattern='cont', x0=25, y0=-30, sd0=7, u=0, v=1, rad_spd=0.3,rad_width=4)
        title = 'cont'


        return dff1, title
        #dff1= create_pattenrs(N=60, M=60, frames=70, pattern='cont', x0=35, y0=-20, sd0=8, u=0, v=1,rad_spd=0.3, rad_width=4)

    if type == 'cortex':
        x = scipy.io.loadmat("/Users/arielrom/Desktop/תואר שני 2/Thesis/AnalyzedData/Early Tryings/spont_4000f_MMStack_Pos0.mat")
        #x = scipy.io.loadmat("/Users/arielrom/Desktop/תואר שני 2/Thesis/AnalyzedData/Early Tryings/spont_higherled_4000f_MMStack_Pos0_1.mat")
        mat_data = MatlabToDff(x)
        mat_data.enhance()
        dff1 = mat_data.dff[:,:,426:511] ##(426,511) ##(0,450) , (200,400)
        title = 'Cortex'

        return decrease_frame_rate(dff1, 25, 10) , title

    if type == 'retina':
        video_path = "/Users/arielrom/Desktop/תואר שני 2/Thesis/AnalyzedData/TifConvert/elife-81983-video1.mp4"
        mp4_data = MP4ToDff(video_path)
        mp4_data.mp4_video_to_numpy_gray()

        mp4_data.dff = mp4_data.dff[:,:,1100:1350] #1100,1400   ,  530,710
        mp4_data.dff = normalize_data(mp4_data.dff)
        title = 'Retina'

        return mp4_data.dff , title

    if type == 'spiral':
        video_path = "/Users/arielrom/Desktop/תואר שני 2/Thesis/AnalyzedData/spiral.avi"
        mp4_data = MP4ToDff(video_path)
        mp4_data.mp4_video_to_numpy_gray()


        mp4_data.dff = mp4_data.dff[:,:,1200:1380] #1100,1400   ,  530,710
        mp4_data.dff = normalize_data(mp4_data.dff)
        title = 'Spiral Wave'
        return mp4_data.dff , title

    if type =='SD':
        video_path = "/Users/arielrom/Desktop/תואר שני 2/Thesis/AnalyzedData/FullCode/SD_WAVE.avi"
        mp4_data = MP4ToDff(video_path)

        mp4_data.mp4_video_to_numpy_gray()
        mp4_data.dff = mp4_data.dff[100:800,100:800,0:600]
        mp4_data.normalize_data()

        dff1 = mp4_data.dff
        title = 'Spreading Depression'

        return apply_lowpass_filter(dff1, 30, 10) , title






class MatlabToDff:
    def __init__(self,data):
        self.dff = data['dff_data']
        self.N = data['dff_data'].shape[0]
        self.M = data['dff_data'].shape[1]
        self.frames = data['dff_data'].shape[2]

    ## enhance Widefield CA imaging output
    def enhance (self):

        def init(z):
            return 1 * z * (z != -1) + 0 * z * (z == -1)
        for t in range(self.frames):
            self.dff[:, :, t] = init(self.dff[:, :, t])

        self.dff[:, :, ] = normalize_data(self.dff[:, :, ])


class MP4ToDff:
    def __init__(self,mp4):
        self.mp4 = mp4
        self.avi = mp4
        self.dff = np

    def mp4_video_to_numpy_gray(self):
        # Open the MP4 video file
        cap = cv2.VideoCapture(self.avi)

        # Get the video properties
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create an empty NumPy array to store the grayscale video frames
        video_array_gray = np.zeros((num_frames, height, width), dtype=np.uint8)

        # Read each frame, convert to grayscale, and store it in the array
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            video_array_gray[i] = frame_gray

        # Release the video capture object
        cap.release()

        self.dff = np.transpose(video_array_gray, (1, 2, 0))
        self.N = self.dff.shape[0]
        self.M = self.dff.shape[1]


class PreDataProcessing:
    def __init__(self,data):
        self.N = data.shape[0]
        self.M = data.shape[1]
        self.frames = data.shape[2]
        self.dff = data

    def resize(self , m ,n):
        resized_dff = np.zeros((n, m, self.frames))
        for i in range (self.frames):
            resized_dff[:,:,i] = resize(self.dff[:,:,i], m, n)
        self.dff = resized_dff
        self.N = resized_dff.shape[0]
        self.M = resized_dff.shape[1]

    def filter(self,fil,sigma,kernel):
        ## filter = 'gaussian' , 'bilateral' , 'average' , 'normalize'
        filtered_dff = np.zeros((self.dff.shape[0],self.dff.shape[1], self.dff.shape[2]))
        for i in range (self.dff.shape[2]):
            filtered_dff[:,:,i] = Filter(self.dff[:,:,i], fil = str(fil), sigma = sigma, kernel = kernel)
        self.dff = filtered_dff

    def add_noise(self,std):
        for t in range (self.frames):
            self.dff[:,:,t] += np.random.normal(0, std, size=(self.N, self.M))


class FlowAnalyze:
    def __init__(self, data):
        self.dff = data.dff
        self.N = data.dff.shape[0]
        self.M = data.dff.shape[1]
        self.frames = data.dff.shape[2]

        self.flows = []

        self.phase_space = np.zeros((self.N , self.M , self.frames-2 ,2))
        self.sum_phase_space = np.zeros((self.N , self.M , self.frames-2 ,2))
        self.sum_phase_space_norm = np.zeros((self.N , self.M , self.frames-2 ,2))

        self.restricted_hessian_values = np.zeros(((self.N , self.M , self.frames)))

        self.Flow = True

    def horn_schunck_flow(self , alpha, num_iter):
        for i in range(self.frames - 1):
            image1 = self.dff[:, :, i]
            image2 = self.dff[:, :, i+1]

            flow = horn_schunck(image1, image2, alpha, num_iter)
            self.flows.append(flow)

    def normalize_flow(self):
        norms = []
        for flow in self.flows:
            # Calculate the squared norm along the last axis (axis=-1) to get the Euclidean norm
            norm = np.linalg.norm(flow, axis=-1) ** 2  # Squaring for the squared Euclidean norm
            norms.append(norm)

        all_nonzero_values = []
        for norm_array in norms:
            flattened_array = norm_array.flatten()


            nonzero_values = flattened_array[flattened_array != 0]             # Filter non-zero values and append them to the list
            all_nonzero_values.extend(nonzero_values)

        # Sort the list of non-zero values based on their magnitude
        sorted_nonzero_values = sorted(all_nonzero_values)
        index_999_percentile = int(len(sorted_nonzero_values) * 0.999)

        # Extract the value at the 99.9th percentile
        max_value = sorted_nonzero_values[index_999_percentile]

        for idx, flow in enumerate(self.flows):
            self.flows[idx] = flow / max_value

    def calculate_map(self, eps):
        def check_div_theo(vector_field,tolerance = eps):
            # Get the dimensions of the vector field
            N, M, _ = vector_field.shape

            # Calculate the divergence using numpy.gradient()   , Notice that the axis are invert
            grad_x = np.gradient(vector_field[:, :, 0], axis=1)  # Gradient along the x-axis
            grad_y = np.gradient(vector_field[:, :, 1], axis=0)  # Gradient along the y-axis
            divergence = grad_x + grad_y

            def calculate_flux(f_x, f_y):
                flux_field = np.zeros_like(f_x)

                for i in range(1, flux_field.shape[0] - 1):
                    for j in range(1, flux_field.shape[1] - 1):
                        ## Create a 3x3 rectangle around each pixel and calculate its flux
                        flux_up = (f_y[i + 1, j + 1] + f_y[i + 1, j] + f_y[i + 1, j - 1]) / 3
                        flux_down = (- f_y[i - 1, j + 1] - f_y[i - 1, j] - f_y[i - 1, j - 1]) / 3

                        flux_right = (f_x[i + 1, j + 1] + f_x[i, j + 1] + f_x[i - 1, j + 1]) / 3
                        flux_left = (- f_x[i + 1, j + 1] - f_x[i, j + 1] - f_x[i - 1, j + 1]) / 3

                        flux_field[i, j] = flux_up + flux_down + flux_right + flux_left

                return flux_field

            flux = calculate_flux(vector_field[:, :, 0], vector_field[:, :, 1])

            div_theo_check = flux - divergence

            return np.where(abs(div_theo_check) > tolerance, 10, div_theo_check)

        for i in range(1, self.frames - 1):
            # Compute temporal gradient
            dA_dt = self.dff[:, :, i + 1] - self.dff[:, :, i]


            self.phase_space[:, :, i - 1, 0] = np.abs(dA_dt) * self.flows[i][:, :, 0]
            self.phase_space[:, :, i - 1, 1] = np.abs(dA_dt) * self.flows[i][:, :, 1]

            self.sum_phase_space[:, :, i - 1, 0] = np.sum ( self.phase_space[:, :, :, 0] , axis = 2)
            self.sum_phase_space[:, :, i - 1, 1] = np.sum ( self.phase_space[:, :, :, 1] , axis = 2)

        # Normalize phase space data
        self.sum_phase_space_norm = normalize_data(self.sum_phase_space)

        ## Remove pixel not fulfilling the divergence theorem
        matrix = check_div_theo(self.sum_phase_space_norm[:, :, -1, :])
        for t in range( self.frames-2) :
            self.sum_phase_space_norm[:,:,t,0] = np.where(matrix == 10, 0, self.sum_phase_space_norm[:,:,t,0])
            self.sum_phase_space_norm[:,:,t,1] = np.where(matrix == 10, 0, self.sum_phase_space_norm[:,:,t,1])


class Display:
    def __init__(self, data):
        self.dff = data.dff
        self.N = data.dff.shape[0]
        self.M = data.dff.shape[1]
        self.frames = data.dff.shape[2]

        try:
            if data.Flow :
                self.flows = data.flows
                self.phase_space = data.phase_space
                self.sum_phase_space = data.sum_phase_space
                self.sum_phase_space_norm = data.sum_phase_space_norm

                self.restricted_hessian_values = np.zeros(((self.N + 1, self.M + 1, self.frames)))
        except AttributeError:
            pass

        original_blues = cm.get_cmap('Blues')
        new_colors = np.vstack((np.array([1, 1, 1, 1]), original_blues(np.linspace(0, 1, 256))))  # Create a new colormap that starts with white and then includes the original Blues colors
        self.color_map = ListedColormap(new_colors, name='WhiteBlues')  # Assign to instance attribute
        self.n = n
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        self.iter = iterations

    def plot_data(self):
        fig, ax = plt.subplots()
        def anima(i):
            ax.cla()
            ax.imshow(self.dff[:, :, i], cmap=self.color_map, vmin=0, vmax=1)#, vmin=0, vmax=1)
            ax.set_title(r" Data plot")
            ax.axis('off')

            return ax
        ani = animation.FuncAnimation(fig, func=anima, frames=self.frames-1, blit=False, interval=100)
        image1 = ax.imshow(self.dff[:,:,0], cmap=self.color_map, vmin=0, vmax=1)
        plt.colorbar(image1)

        #ani.save('SD.gif', writer='pillow')

        plt.show()

    def plot_data_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = np.arange(self.dff.shape[0])
        y = np.arange(self.dff.shape[1])
        X, Y = np.meshgrid(x, y)

        def anima(i):
            ax.cla()
            Z = self.dff[:, :, i]
            ax.plot_surface(X, Y, Z, cmap='plasma', vmin=0, vmax=1)
            ax.set_title(r"2 Discrete Gaussians , $\Delta$X = 2.5$\sigma$ , $\Delta$T = 2$\sigma$")
            ax.set_zlim(0,1)
            ax.text2D(0.85, 0.02, f"{i / 10} [sec]", transform=ax.transAxes, ha='center', color='white', fontsize=12)
            return ax

        ani = animation.FuncAnimation(fig, func=anima, frames=self.frames - 1, blit=False, interval=300)

        ani.save('2Gaussians_3D.gif', writer='pillow')

        plt.show()


    def plot_data_div(self):
        fig, (ax,ax2) = plt.subplots(nrows=1 , ncols=2, figsize=(20, 10))
        def anima(i):
            ax.cla()
            ax.imshow(self.dff[:, :, i], cmap=self.color_map, vmin=0, vmax=1)#, vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis('off')

            #ax2.cla()


            return ax

        A = check_div_theo(self.sum_phase_space_norm[:, :, -1, :])
        ax2.imshow(check_div_theo(self.sum_phase_space_norm[:, :, -1, :]), cmap='BrBG', vmin=-0.5, vmax=0.5)
        ax2.set_title(r"Divergence Theorem Check")
        ax2.axis('off')

        ani= animation.FuncAnimation(fig, func=anima, frames=self.frames-2, blit=False, interval=100)
        image1 = ax.imshow(self.dff[:,:,0], cmap=self.color_map, vmin=0, vmax=1)
        plt.colorbar(image1)

        image2 = ax2.imshow(A, cmap='BrBG', vmin=-0.5, vmax=0.5)
        plt.colorbar(image2)

        #ani.save('2 Gaussians.gif', writer='pillow')

        plt.show()


    def spatial_connections_video(self ,space ,scale):
        fig, ((ax,ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))

        def anima(i):
            ax.cla()
            plot_quiver(ax, self.flows[i], spacing=space, scale=scale, color='black')
            ax.imshow(self.dff[:, :, i], cmap=self.color_map, vmin=0, vmax=1)
            ax.set_ylim(self.dff.shape[1], 0)
            ax.axis("off")
            ax.set_title(fr"{self.N}x{self.M}x{self.frames} , {self.iter} iter ,{title}")


            return [ax,ax2,ax3,ax4]

        plot_quiver(ax2, self.sum_phase_space[:, :, -1, :], spacing=1, scale=scale, color='black')
        ax2.set_ylim(self.dff.shape[1], 0)
        ax2.set_xlim(0, self.dff.shape[1])

        ax2.text(0.5, 1.1, f"Algo output\n$\u03B1$={self.alpha}, $\u03B5$={self.epsilon} ",
                 horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        ax2.axis("off")

        Waveness = np.zeros((self.N , self.M ,4))
        pad_width = 4

        pad_phase_space = np.pad(self.sum_phase_space,pad_width=((pad_width, pad_width), (pad_width, pad_width), (0, 0),(0,0)), mode='constant', constant_values=0)

        for h in range(self.n, self.N + self.n-2):
            for j in range(self.n, self.M +self.n-2):
                vector_field = pad_phase_space[h - self.n:h + self.n+1, j - self.n:j + self.n+1, -1, :]

                node_counts, theo_counts = nodes_connections(vector_field,theta=self.theta, plot='no')
                max_node_counts, max_theo_counts = nodes_connections(vector_field,theta=180, plot='no')

                if max_node_counts[4] == 0:
                    result = 0
                else:
                    result = np.round(node_counts[4] / max_node_counts[4], 2)

                # Initialize sums
                sum_vx = 0
                sum_vy = 0
                for row in vector_field:
                    for vector in row:
                        sum_vx += vector[0]
                        sum_vy += vector[1]

                # Calculate averages
                avg_vx = sum_vx / ((2*self.n)**2)
                avg_vy = sum_vy / ((2*self.n)**2)

                angles = np.arctan2(avg_vx, avg_vy)
                normalized_angles = (angles + np.pi) / (2 * np.pi)
                color = mcolors.hsv_to_rgb([normalized_angles, 1, 1])  # Saturation and Value are 1

                Waveness[h-self.n , j-self.n ,0] = color[0]
                Waveness[h-self.n , j-self.n ,1] = color[1]
                Waveness[h-self.n , j-self.n ,2] = color[2]
                Waveness[h - self.n, j - self.n, 3] = result * np.sqrt( (self.sum_phase_space_norm[h - self.n, j - self.n, -1, 0]) ** 2 + (self.sum_phase_space_norm[h - self.n, j - self.n, -1, 1]) ** 2)
        ax3.set_title(f'Data degree after $\u03B8$={self.theta}° , n={self.n}')

        ax3.imshow(Waveness)
        norm = plt.Normalize(0, 2 * np.pi)
        cmap = plt.cm.hsv


        # Create a ScalarMappable and initialize a data array with the full range of angles
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # Add the colorbar to the figure
        cbar = plt.colorbar(sm, ax=ax3, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Direction (angle)')
        cbar.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        cbar.set_ticklabels(['0', 'π/2', 'π', '3π/2', '2π'])



        image1 = ax.imshow(self.dff[:, :, 0], cmap=self.color_map, vmin=0, vmax=1)
        plt.colorbar(image1, orientation='horizontal')

        ax3.set_xticks([])
        ax3.set_yticks([])

        flattened_values = Waveness[:,:,3].flatten()   ## Waveness[3] is the wavness parameter
        ax4.hist(flattened_values , bins = 100, range = (0,1))
        ax4.set_ylim(0,100)
        ax4.set_title ( 'Histogram of Waveness values')

        ani = animation.FuncAnimation(fig, func=anima, frames=self.frames-2, blit=False, interval=100)
        ani.save('cortex .gif', writer='pillow')

        plt.show()


    def curl_divergence(self ,space ,scale):
        fig, ( (ax,ax2),(ax3,ax4) ) = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))

        def curl_2d(vector_field):
            Fx = vector_field[:, :, 0]
            Fy = vector_field[:, :, 1]

            dFy_dx = np.gradient(Fy, axis=1)  # partial derivative of Fy with respect to x
            dFx_dy = np.gradient(Fx, axis=0)  # partial derivative of Fx with respect to y

            curl_field = dFy_dx - dFx_dy

            return curl_field

        def divergence_2d(vector_field):
            Fx = vector_field[:, :, 0]
            Fy = vector_field[:, :, 1]

            dFx_dx = np.gradient(Fx, axis=1)  # partial derivative of Fx with respect to x
            dFy_dy = np.gradient(Fy, axis=0)  # partial derivative of Fy with respect to y

            divergence_field = dFx_dx + dFy_dy

            return divergence_field

        def anima(i):
            ax.cla()
            plot_quiver(ax, self.flows[i], spacing=space, scale=scale, color='black')
            ax.imshow(self.dff[:, :, i], cmap=self.color_map, vmin=0, vmax=1)
            ax.set_ylim(self.dff.shape[1], 0)
            ax.axis("off")
            ax.set_title(fr"{self.N}x{self.M}x{self.frames} , {self.iter} iter ,{title}")

            ax2.cla()
            plot_quiver(ax2, self.sum_phase_space[:, :, -1, :], spacing=space, scale=scale, color='black')
            ax2.set_ylim(self.dff.shape[1], 0)
            ax2.set_xlim(0,self.dff.shape[1])

            #ax2.text(0.5, 1.1, f"Algo output\n$\u03B1$={self.alpha}, $\u03B5$={self.epsilon} ", horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
            ax2.text(0.5, 1.1, f"Algo output\n$\u03B1$={self.alpha}, no $\u03B5$ clean ", horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)

            ax2.axis("off")


            ax3.cla()
            ax3.imshow(curl_2d(self.sum_phase_space[:, :, i, :]), cmap='RdBu', vmin=-0.7, vmax=0.7)
            ax3.set_ylim(self.dff.shape[1], 0)
            ax3.set_xlim(0,self.dff.shape[1])

            ax3.set_title(f'Curl of accumlated algo output')
            ax3.axis("off")

            ax4.cla()
            ax4.imshow(divergence_2d(self.sum_phase_space[:, :, i, :]), cmap='BrBG', vmin=-0.7, vmax=0.7)
            ax4.set_ylim(self.dff.shape[1], 0)
            ax4.set_xlim(0,self.dff.shape[1])

            ax4.set_title(f'Divergence of accumlated algo output')
            ax4.axis("off")

            return [ax,ax2,ax3]

        image1 = ax3.imshow(curl_2d(self.sum_phase_space[:, :, 0, :]), cmap='RdBu', vmin=-0.7, vmax=0.7)
        #cax = fig.add_axes([0.675, 0.2, 0.23, 0.04])  # Adjust the position and size as needed
        plt.colorbar(image1)#, cax=cax, orientation='horizontal')

        image2 = ax4.imshow(divergence_2d(self.sum_phase_space[:, :, 0, :]), cmap='BrBG', vmin=-0.7, vmax=0.7)
        #cax = fig.add_axes([0.675, 0.2, 0.23, 0.04])  # Adjust the position and size as needed
        plt.colorbar(image2)#, cax=cax, orientation='horizontal')


        ani = animation.FuncAnimation(fig, func=anima, frames=self.frames-2, blit=False, interval=100)
        #ani.save('Plane Curl.gif', writer='pillow')

        plt.show()



def check_div_theo(vector_field, epsilon=epsilon):
    N, M, _ = vector_field.shape

    # Calculate the divergence using numpy.gradient()   , Notice that the axis are invert
    grad_x = np.gradient(vector_field[:, :, 0], axis=1)  # Gradient along the x-axis
    grad_y = np.gradient(vector_field[:, :, 1], axis=0)  # Gradient along the y-axis
    divergence = grad_x + grad_y

    def calculate_flux(f_x, f_y):
        flux_field = np.zeros_like(f_x)
        for i in range(1, flux_field.shape[0] - 1):
            for j in range(1, flux_field.shape[1] - 1):
                flux_up = ( f_y[i + 1, j + 1] + f_y[i + 1, j ] + f_y[i + 1, j - 1] ) / 3
                flux_down =  ( - f_y[i - 1, j + 1] - f_y[i - 1, j] - f_y[i - 1, j - 1] ) / 3

                flux_right = ( f_x[i + 1, j + 1] + f_x[i, j + 1] + f_x[i - 1, j + 1] ) /3
                flux_left =  ( - f_x[i + 1, j + 1] - f_x[i, j + 1] - f_x[i - 1, j + 1] ) / 3
                flux_field[i, j] = flux_up + flux_down + flux_right + flux_left


        return flux_field

    flux = calculate_flux(vector_field[:, :, 0], vector_field[:, :, 1])


    div_theo_check = flux - divergence
    return  np.where(abs(div_theo_check) > epsilon, 10, div_theo_check)



dff1 ,title = data('2 gaussian')

data = PreDataProcessing(dff1)
#data.resize(100,100)

###RETINA####
#data.resize(200,200)
#data.dff = data.dff[70:170,25:125,:]

data = FlowAnalyze(data)
data.horn_schunck_flow(alpha=alpha, num_iter=iterations )
data.normalize_flow()
data.calculate_map(eps=epsilon)

display=Display(data)


#display.plot_data_div()
display.spatial_connections_video(space = 1 , scale = 2 )
#display.curl_divergence(space = 1 , scale = 10 )


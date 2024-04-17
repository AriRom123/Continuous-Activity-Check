import numpy as np
import math


def plot_quiver(ax, flow, spacing, margin=0, **kwargs):
    """Plots less dense quiver field.

    Args:
        ax: Matplotlib axis
        flow: motion vectors
        spacing: space (px) between each arrow in grid
        margin: width (px) of enclosing region without arrows
        kwargs: quiver kwargs (default: angles="xy", scale_units="xy")
    """
    w ,h , *_ = flow.shape

    nx = int(math.ceil((w - 2 * margin) / spacing))
    ny = int(math.ceil((h - 2 * margin) / spacing))

    x = np.linspace(margin + 0.5*spacing, w - margin - 0.5*spacing , nx, dtype=np.int64)
    y = np.linspace(margin + 0.5*spacing, h - margin - 0.5*spacing , ny, dtype=np.int64)
    #print(x,y)



    #y = y[::-1]  # Invert the order of y values to flip the y-axis

    flow = flow[np.ix_(x,y)]
    u =  flow[:, :, 0]
    v =  flow[:, :, 1]


    xx, yy = np.meshgrid(x, y)
    #print(xx)

    kwargs = {**dict(angles="xy", scale_units="xy"), **kwargs}
    ax.quiver(xx, yy, u, v, **kwargs)

    ax.set_xlim([-0.5, w -0.5])
    ax.set_ylim([-0.5, h - 0.5])
    ax.set_aspect('equal')

def calculate_frenet_serret(points):
    # Calculate the tangent vector at each point using finite differences
    tangent_vectors = np.diff(points, axis=0)

    # Normalize tangent vectors
    tangent_vectors = tangent_vectors/ np.linalg.norm(tangent_vectors, axis=1, keepdims=True)

    # Calculate the normal vectors at each point unsing finite differences
    normal_vectors = np.diff(tangent_vectors, axis=0)
    # Delete epsilon values
    ##epsilon = 0.000001
    ##normal_vectors [ abs(normal_vectors) <= epsilon] = 0


    # Calculate the curvature at each point
    curvature = np.linalg.norm(np.diff(tangent_vectors, axis=0), axis=1) / np.linalg.norm(tangent_vectors[:-1], axis=1)

    # Return the tangent and normal vectors as separate arrays
    return tangent_vectors, normal_vectors , curvature

def trajectory (dff,flows, x0,y0,T):
    length = dff.shape[2]

    x_traj = []
    y_traj = []

    for dT in range (1,T):
    #Start at time dT

        x_traj_dT = [x0 for i in range(dT)]
        y_traj_dT = [y0 for i in range(dT)]

        U_dT=[]
        V_dT=[]
        for i in range(dT-1,length-1):

            x_pos = x_traj_dT[-1]
            y_pos = y_traj_dT[-1]
            x_int = int(math.floor(x_pos))
            y_int = int(math.floor(y_pos))
            x_frac = x_pos - x_int
            y_frac = y_pos - y_int
            if y_int >= dff.shape[1] - 1 or x_int >= dff.shape[0] - 1 or y_int <= 0 or x_int <= 0:      ## Make sure to not cross boundaries
                u , v = 0 ,0
            else:
                u = (1 - x_frac) * (1 - y_frac) * flows[i][y_int, x_int, 0] + \
                    x_frac * (1 - y_frac) * flows[i][y_int, x_int + 1, 0] + \
                    (1 - x_frac) * y_frac * flows[i][y_int + 1, x_int, 0] + \
                    x_frac * y_frac * flows[i][y_int + 1, x_int + 1, 0]
                v = (1 - x_frac) * (1 - y_frac) * flows[i][y_int, x_int, 1] + \
                    x_frac * (1 - y_frac) * flows[i][y_int, x_int + 1, 1] + \
                    (1 - x_frac) * y_frac * flows[i][y_int + 1, x_int, 1] + \
                    x_frac * y_frac * flows[i][y_int + 1, x_int + 1, 1]
                U_dT.append(u)
                V_dT.append(v)
                x_traj_dT.append(x_pos + u)
                y_traj_dT.append(y_pos + v)
        x_traj.append(x_traj_dT)
        y_traj.append(y_traj_dT)

    points = np.array ( [(x_traj_dT[i],y_traj_dT[i]) for i in range (len(x_traj_dT))] )

    tangent_vectors, normal_vectors , curvature = calculate_frenet_serret(points)
    return x_traj , y_traj , points , tangent_vectors , normal_vectors , curvature

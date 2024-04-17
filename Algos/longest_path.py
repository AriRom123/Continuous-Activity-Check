import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy.linalg import norm
import math


def angle_between_vectors(v1, v2):
    v1_norm = norm(v1)
    v2_norm = norm(v2)

    if v1_norm == 0 or v2_norm == 0:
        return np.nan  # Return NaN if either vector has zero magnitude to avoid division by zero

    cosine_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
    # Check for edge cases where the value might be slightly outside the valid range due to precision issues
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.arccos(cosine_angle)


def count_nodes_by_connections(nodes_connections):
    connection_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for node, connections in nodes_connections.items():
        if connections <= 4:
            connection_counts[connections] += 1

    return connection_counts


def nodes_connections(vector_field, theta,plot  ):
    M, N, _ = vector_field.shape

    G = nx.DiGraph()
    all_edges = nx.DiGraph()  # Graph to store all edges

    nodes_connections = {}  # Dictionary to store connections for each node
    all_connections = {}


    for i in range(M):
        for j in range(N):
            current_node = (i, j)
            G.add_node(current_node)
            all_edges.add_node(current_node)

            neighbors = [
                (i - 1, j),
                (i + 1, j),
                (i, j - 1),
                (i, j + 1),
            ]

            connections = 0  # Counter for connections of the current node
            all_con = 0

            for ni, nj in neighbors:
                if 0 <= ni < M and 0 <= nj < N:
                    neighbor_node = (ni, nj)
                    angle = angle_between_vectors(vector_field[i, j], vector_field[ni, nj]) * 180 / math.pi
                    dot_product = np.dot(vector_field[i, j], vector_field[ni, nj])

                    if abs(angle) < theta:
                        G.add_edge(current_node, neighbor_node, weight=dot_product, Angle=angle)
                        connections += 1  # Increment the connection counter for the current node

                    if not np.isnan(angle) :
                        all_edges.add_node(current_node)
                        all_edges.add_edge(current_node, neighbor_node, weight=dot_product, Angle=angle)
                        all_con+=1


            nodes_connections[current_node] = connections
            all_connections[current_node] = all_con


    node_counts = count_nodes_by_connections(nodes_connections)
    theo_counts = count_nodes_by_connections(all_connections)


    filtered_edges = [(u, v) for u, v, d in G.edges(data=True) if abs(d['Angle']) < theta]
    filtered_graph = G.edge_subgraph(filtered_edges)

    theoretical_edges = [(u, v) for u, v, d in all_edges.edges(data=True)]
    theoretical_graph = all_edges.edge_subgraph(theoretical_edges)

    pos = {node: node for node in G.nodes()}
    all_pos = {node: node for node in all_edges.nodes()}
    if plot =='yes':
        fig, ax = plt.subplots(figsize=(6, 6))
        X, Y = np.meshgrid(np.arange(N), np.arange(M))
        ax.set_title('Vector Field ')
        ax.set_xlim(-0.25, M-0.5)
        ax.set_ylim(-0.25, N-0.5)

        nx.draw_networkx_edges(theoretical_graph, {k: (x, M - y - 1) for k, (y, x) in all_pos.items()},node_size=50, edge_color='tomato', alpha=0.15, width=2, ax=ax, arrows=False)
        nx.draw_networkx_nodes(theoretical_graph, {k: (x, M - y - 1) for k, (y, x) in all_pos.items()},node_size=30,ax=ax ,node_color='black')

        # Plot the reversed filtered graph
        nx.draw(filtered_graph.reverse(), {k: (x, M - y - 1) for k, (y, x) in pos.items()},
                with_labels=False, node_size=30, node_color='black', edge_color='tomato',
                alpha=1, width=4, font_weight='bold', ax=ax, arrows=False)
        ax.quiver(X, M - Y - 1, vector_field[:, :, 0], -vector_field[:, :, 1], scale=30 )  # Invert y-axis
        #ax3.set_title(r'Filtered Graph, $\theta$ = {} [degrees]'.format(theta))


        plt.tight_layout()

       # plt.show()

    return node_counts , theo_counts



# Example usage:
vector_field=(np.random.rand(4,4,2)-0.1)/2
vector_field[vector_field<0.4]=0
#vector_field = np.array([[[0, 0], [0, 0], [0.33142296, 0.52608871],[0, 0]],
#    [[0, 0], [0.43258202, 0.21398492], [0.33142296, 0.52608871],[0.33142296, 0.52608871]],
#    [[0.47284963, -0.20674897], [0.53551432, 0.23384063], [0.22213769, 0.35897794],[0.33142296, 0.52608871]],
#    [[0.56552441, -0.19352592], [-0.41081992, 0.4015286], [-0.45122815, -0.361123],[0.33142296, 0.52608871]]])

#nodes_connections(vector_field, theta=30)



#print('average cluster: ',x)

#find_longest_path_2d(vector_field, theta=10)

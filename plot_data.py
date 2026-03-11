import matplotlib.pyplot as plt
import numpy as np

def plot_gaussian_dataset(data1, data2, title, color="blue"):
    """
    Plots a 2D array of Gaussian samples.
    Expects data in an N x 2 matrix format.
    """
    x1 = data1[:, 0]
    y1 = data1[:, 1]

    x2 = data2[:, 0]
    y2 = data2[:, 1]

    x = np.append(x1, x2)
    y = np.append(y1, y2)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x1, y1, color="red", s=1, alpha=0.05)
    ax.scatter(x2, y2, color=color, s=1, alpha=0.05)
    ax.set_aspect('equal')
    
    # # ax.axline(boundary_x, boundary_y, color="red")
    x_line = np.linspace(-2.5, 8, 100)
    y_line = -x_line + 4.718
    ax.plot(x_line, y_line, color="red")

    ax.set_title(title)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(title)

# def plot_classifications(classifications, title):
#     """
#     Plots a 2D array of Gaussian samples.
#     Expects data in an N x 2 matrix format.
#     """
#     x_red = []
#     y_red = []

    

#     for key, value in classifications:

    
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.scatter(x, y, color=color, s=1, alpha=0.05)
#     ax.set_aspect('equal')
    
#     ax.set_title(title)
#     ax.set_xlabel("X Axis")
#     ax.set_ylabel("Y Axis")
#     ax.grid(True, linestyle='--', alpha=0.5)
    
#     plt.savefig(title)
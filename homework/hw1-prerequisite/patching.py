import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def get_patch(mask, patch_shape):
    """This is for generate corresponding patches.

    Args:
        mask (numpy.ndarray): A circle mask array.
        patch_shape (numpy.ndarray):
            The patch shape will larger than the bounding box shape.

    Returns:
        patch (numpy.ndarray)

    """
    xmin, xmax, ymin, ymax = np.inf, -1, np.inf, -1
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if (mask[i][j]):
                if(i < xmin): xmin = i
                if(i > xmax): xmax = i
                if(j < ymin): ymin = j
                if(j > ymax): ymax = j
    circle = mask[xmin:xmax+1, ymin:ymax+1]
    
    patch = np.zeros(patch_shape.shape)
    
    x_m = patch_shape.shape[0] // 2
    x_sm = circle.shape[0] // 2
    
    y_m = patch_shape.shape[1] // 2
    y_sm = circle.shape[1] // 2
    
    patch[x_m-x_sm : x_m-x_sm+circle.shape[0], y_m-y_sm : y_m-y_sm+circle.shape[1]] = circle[:,:]
    
    
    return patch
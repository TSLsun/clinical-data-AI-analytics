# import matplotlib
# matplotlib.use("AGG")
import matplotlib.pyplot as plt

def plot_hu_distribution(hu_image):
    fig = plt.figure(figsize=(15, 7.5))
    plt.hist(hu_image.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()
    
def plot_ct_scan(image):
    rows = 8
    cols = 8
    interval_z = image.shape[0] // (rows * cols)
    fig, plots = plt.subplots(rows, cols, figsize=(45, 45))
    plt.subplots_adjust(
        wspace=.05,
        hspace=.05,
        left=0,
        right=1,
        top=1,
        bottom=0
    )
    z = 0
    for i in range(rows):
        for j in range(cols):
            if z <= len(image):
                plots[i, j].axis('off')
                plots[i, j].imshow(image[z], cmap='gray')
                z += interval_z
import os
import glob
import numpy as np 
import pydicom as dicom
import nibabel as nib
# import matplotlib
# matplotlib.use("AGG")
import matplotlib.pyplot as plt

from scipy import ndimage as scndi

from sklearn.cluster import KMeans

from skimage import morphology
from skimage import measure
from skimage.transform import resize
from skimage.segmentation import clear_border
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.filters import roberts, sobel


## Load image and label

def load_dicom_volume(src_dir, suffix='*.dcm'):
    """Load DICOM volume and get meta data.
    """
    encode_name = src_dir.split('/')[-1]
    # Read dicom files from the source directory
    # Sort the dicom slices in their respective order by slice location
    dicom_scans = [dicom.read_file(sp) for sp in glob.glob(os.path.join(src_dir, suffix))]
    # dicom_scans.sort(key=lambda s: float(s.SliceLocation))
    dicom_scans.sort(key=lambda s: float(s[(0x0020, 0x0032)][2]))

    # Convert to int16, should be possible as values should always be low enough
    # Volume image is in z, y, x order
    volume_image = np.stack([ds.pixel_array for ds in dicom_scans]).astype(np.int16)

    # Get data info
    spacing = [float(dicom_scans[0].SliceThickness)] + list(map(float, dicom_scans[0].PixelSpacing))
    patient_position = list(map(float, dicom_scans[0].ImagePositionPatient))
    rescaleIntercept = int(dicom_scans[0].RescaleIntercept)
    rescaleSlope =  int(dicom_scans[0].RescaleSlope)
    info_dict = {"PatientPosition": patient_position, 'Spacing': spacing,
                 'RescaleIntercept': rescaleIntercept, 'RescaleSlope': rescaleSlope}
    
    return encode_name, volume_image, info_dict

def load_label(label_fpath, transpose=False):
    encode_name = label_fpath[-39: -7]
    label_data = nib.load(label_fpath)
    label_array = label_data.get_fdata()
    if transpose:
        label_array = np.transpose(label_array, axes=(2,1,0))
    return encode_name, label_array 

## preprocess

def hounsfield_unit_tranformation(image, slope, intercept):
    """Transfer dicom image's data to Hounsfield units (HU)
    """
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    # image[image == -2000] = 0

    # Transform type
    slope = float(slope)
    intercept = float(intercept)
    image = image.astype(np.float64)

    # Convert to Hounsfield units (HU)
    image = (slope * image).astype(np.int16) + np.int16(intercept)
    return image.astype(np.int16)

def resample(image, original_spacing, new_spacing=[1, 1, 1]):
    """ Resample images to specific new spacing.
    
    Reference:
        https://stackoverflow.com/questions/13242382/resampling-a-numpy-array-representing-an-image
        http://scipy.github.io/devdocs/generated/scipy.ndimage.zoom.html#scipy.ndimage.zoom
    """
    # Transfer type
    original_spacing = np.array(original_spacing)
    new_spacing = np.array(new_spacing)
    
    # Calculate resize factor
    resize_factor = original_spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = original_spacing / real_resize_factor
    
    # Interpolate
    image = scndi.zoom(image, real_resize_factor, order=0, mode='nearest')
    return image, new_spacing

def window_normalization(image, lower_th=-1000., upper_th=400., scaling=1.):
    """ Window normalization to 0 ~ scaling
    """
    img_window = np.clip(image, lower_th, upper_th)
    img_window = (img_window - lower_th) / (upper_th - lower_th) * scaling
    return img_window

def zero_center(image, pixel_mean=0.18475):
    image = image - pixel_mean
    return image

#Standardize the pixel values
def make_lungmask(img, display=False):
    """
        ref: https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
    """
    row_size= img.shape[0]
    col_size = img.shape[1]
    
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean
    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
    # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img,np.ones([3,3]))
    dilation = morphology.dilation(eroded,np.ones([8,8]))

    labels = measure.label(dilation) # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0

    #
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    #
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
#     mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
    mask = morphology.dilation(mask,np.ones([40,40])) # one last dilation

    if (display):
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask*img, cmap='gray')
        ax[2, 1].axis('off')
        
        plt.show()
    return mask*img

### another mask approach # will modify original

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None
    
def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image


def get_segmented_lungs(a_slice, plot=False):
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        fig, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image.
    '''
    binary = a_slice < 604
    if plot == True:
        plots[0].axis('off')
        plots[0].set_title('binary image')
        plots[0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].set_title('after clear border')
        plots[1].imshow(cleared, cmap=plt.cm.bone)
    '''
    Step 3: Label the image.
    '''
    label_image = measure.label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].set_title('found all connective graph')
        plots[2].imshow(label_image, cmap=plt.cm.bone)
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in measure.regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].set_title(' Keep the labels with 2 largest areas')
        plots[3].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].set_title('seperate the lung nodules attached to the blood vessels')
        plots[4].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].set_title('keep nodules attached to the lung wall')
        plots[5].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = scndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].set_title('Fill in the small holes inside the binary mask of lungs')
        plots[6].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    a_slice[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].set_title('Superimpose the binary mask on the input image')
        plots[7].imshow(a_slice, cmap=plt.cm.bone)
    return a_slice
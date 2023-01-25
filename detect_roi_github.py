from typing import Tuple

import numpy as np

from skimage.feature import canny
from skimage.morphology import dilation, square
from skimage.measure import label
from matplotlib.patches import Rectangle

import SimpleITK as sitk


class RegionOfInterest():
    
    def __init__(self):
        pass

    @staticmethod
    def detect_roi_dilate_n_crop(sitk_image: sitk.Image,
                      debug: bool = True) -> Tuple[int]:
        image = sitk.GetArrayFromImage(sitk_image)
        image = np.swapaxes(image, 0, -1)
        
        es_slice_index = image.shape[-1] // 2
        ed_slice = image[:, :, 0]
        es_slice = image[:, :, es_slice_index]
        
        diff_image = abs(ed_slice - es_slice)
        edge_image = canny(diff_image, sigma=2.0, low_threshold=0.6, high_threshold=0.96,
                           use_quantiles=True)
        
        edge_image_dilated = dilation(edge_image, square(5))
        
        labels = label(edge_image_dilated)
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        
        temp_top = largestCC.shape[0]
        temp_bottom = 0
        temp_left = largestCC.shape[1]
        temp_right = 0
        for (row_index, row) in enumerate(largestCC):
            for (col_index, col) in enumerate(row):
                if col > 0:
                    # Get the top-most coordinate
                    if row_index < temp_top:
                        temp_top = row_index
                    # Get the bottom-most coordinate
                    if row_index > temp_bottom:
                        temp_bottom = row_index
                    # Get the left-most coordinate
                    if col_index < temp_left:
                        temp_left = col_index
                    # Get the right-most coordinate
                    if col_index > temp_right:
                        temp_right = col_index
        box = [temp_left, temp_right, temp_top, temp_bottom]
        rect = Rectangle((box[0],box[2]),(box[1]-box[0]),(box[3]-box[2]),linewidth=1,edgecolor='r',facecolor='none')

        if debug:
            import matplotlib.pyplot as plt
            
            plt.imshow(ed_slice, cmap='bone')
            plt.title('Passed End Diastolic Image')
            plt.axis('off')
            plt.show()
            plt.close()
            plt.imshow(es_slice, cmap='bone')
            plt.title('Passed End Systolic Image')
            plt.axis('off')
            plt.show()
            plt.close()
            
            plt.imshow(diff_image, cmap='magma')
            plt.title('Difference between ED and ES')
            plt.axis('off')
            plt.show()
            plt.close()
             
            plt.imshow(edge_image, cmap='cubehelix')
            plt.title('Detected Edges on Difference Image')
            plt.axis('off')
            plt.show()
            plt.close()
            
            plt.imshow(edge_image_dilated, cmap='cubehelix')
            plt.title('Dilated Edges Image')
            plt.axis('off')
            plt.show()
            plt.close()
            
            plt.imshow(largestCC, cmap='cubehelix')
            ax = plt.gca()
            ax.add_patch(rect)
            plt.title('Largest Connected Component')
            plt.axis('off')
            plt.show()
            plt.close()
        
        return box

    

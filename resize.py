from typing import List, Union, Tuple

from multiprocessing import Pool

import numpy as np
from scipy import ndimage

from skimage import color
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from sklearn.cluster import KMeans

import SimpleITK as sitk


class Resize():
    
    @staticmethod
    def resample_image(image: sitk.Image, out_spacing: Tuple[float] = (1.0, 1.0, 1.0),
                       out_size: Union[None, Tuple[int]] = None, is_label: bool = False,
                       pad_value: float = 0) -> sitk.Image:
        original_spacing = np.array(image.GetSpacing())
        original_size = np.array(image.GetSize())
        
        if original_size[-1] == 1:
            out_spacing = list(out_spacing)
            out_spacing[-1] = original_spacing[-1]
            out_spacing = tuple(out_spacing)
    
        if out_size is None:
            out_size = np.round(np.array(original_size * original_spacing / np.array(out_spacing))).astype(int)
        else:
            out_size = np.array(out_size)
    
        original_direction = np.array(image.GetDirection()).reshape(len(original_spacing),-1)
        original_center = (np.array(original_size, dtype=float) - 1.0) / 2.0 * original_spacing
        out_center = (np.array(out_size, dtype=float) - 1.0) / 2.0 * np.array(out_spacing)
    
        original_center = np.matmul(original_direction, original_center)
        out_center = np.matmul(original_direction, out_center)
        out_origin = np.array(image.GetOrigin()) + (original_center - out_center)
    
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_size.tolist())
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(out_origin.tolist())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(pad_value)
    
        if is_label:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetInterpolator(sitk.sitkBSpline)
    
        return resample.Execute(image)
    
    
    @staticmethod
    def _get_bounds_and_padding(dimension_image_size: int, dimension_centroid: int,
                                dimension_crop_length: int) -> Tuple[int]:
        
        min_index = dimension_centroid - dimension_crop_length // 2
        pad_min_value = 0
        if min_index < 0:
            pad_min_value = abs(min_index)
            min_index = 0
        
        max_index = min_index + dimension_crop_length
        pad_max_value = 0
        if max_index > dimension_image_size:
            pad_max_value = max_index - dimension_image_size
    
        return min_index, max_index, pad_min_value, pad_max_value
    
    
    @staticmethod
    def crop(image: sitk.Image, centroid: Tuple[int], length: Tuple[int],
             ignore_z_axis: bool = False, padding: float = 0) -> sitk.Image:
        size = image.GetSize()
        
        min_x, max_x, pad_min_x, pad_max_x = Resize._get_bounds_and_padding(size[0],
                                                                            centroid[0],
                                                                            length[0])
        
        min_y, max_y, pad_min_y, pad_max_y = Resize._get_bounds_and_padding(size[1],
                                                                            centroid[1],
                                                                            length[1])
        
        min_z, max_z, pad_min_z, pad_max_z = Resize._get_bounds_and_padding(size[2],
                                                                            centroid[2],
                                                                            length[2])

        if ignore_z_axis:
            pad_min_z = 0
            pad_max_z = 0
            min_z = 0
            max_z = size[-1]
        
        lower_padding = np.asarray([pad_max_x, pad_max_y, pad_max_z]).astype(np.uint32).tolist()
        upper_padding = np.asarray([pad_min_x, pad_min_y, pad_min_z]).astype(np.uint32).tolist()
        padded_image = sitk.ConstantPad(image, upper_padding, lower_padding, padding)
        
        cropped_image = padded_image[min_x: max_x, min_y: max_y, min_z: max_z]
        
        return cropped_image
    
    
    @staticmethod
    def pad(image: sitk.Image, lower_bound: Tuple[int] = [0, 0, 0],
            upper_bound: Tuple[int] = [0, 0, 0], constant: float = 0) -> sitk.Image:
        pad_filter = sitk.ConstantPadImageFilter()

        pad_filter.SetConstant(constant)
        pad_filter.SetPadLowerBound(lower_bound)
        pad_filter.SetPadUpperBound(upper_bound)
        padded_image = pad_filter.Execute(image)
        
        return padded_image
    
    
    

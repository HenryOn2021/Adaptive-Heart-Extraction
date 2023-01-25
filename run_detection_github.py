import os

import numpy as np

import SimpleITK as sitk

import matplotlib.pyplot as plt

from detect_roi_henry import RegionOfInterest
from resize import Resize

import nibabel as nib

def sitk_to_numpy(sitk_image: sitk.Image) -> np.ndarray:
    image = sitk.GetArrayFromImage(sitk_image)
    # Swap the axes when storing as numpy. Sitk stores the axis as z, y, x.
    # So, we switched to x, y, z as it is more common representation
    image = np.swapaxes(image, 0, -1)
    
    return image
    

def detect_roi(path, FolderPath_3D, FolderPath_2D, plot_debug: bool = False) -> None:
    image = sitk.ReadImage(path)
    image_shape = sitk.GetArrayFromImage(image).shape
    image_shape = np.swapaxes(image_shape, 0, -1)
    
    affine = nib.load(path).affine
    header = nib.load(path).header

    roi_algorithm = RegionOfInterest()
    
    image = Resize.resample_image(image)
    rescaled_affine = nib.affines.rescale_affine(affine, image_shape, (1,1,1))
    rescaled_header = header
    rescaled_header['pixdim'][1:4] = [1, 1, 1]
    
    box = roi_algorithm.detect_roi_dilate_n_crop(image,
                                                 debug=plot_debug)
    
    image = sitk.GetArrayFromImage(image)
    image = np.swapaxes(image, 0, -1)
    
    new_img_select_temp = []
    if int(affine[2][2]) > 4:
        for i in range(0, image.shape[-1], int(affine[2][2]) - 1):
            new_img_select_temp.append(i)
        image = image[:,:,(new_img_select_temp)]
    
    cropped_image = image[box[2]:box[3], box[0]:box[1]]
    cropped_image_row = cropped_image.shape[0]
    cropped_image_col = cropped_image.shape[1]
    dimension_diff = abs(cropped_image_row - cropped_image_col)
    pad_before = dimension_diff // 2
    pad_after = dimension_diff - pad_before
    if cropped_image_row > cropped_image_col:
        cropped_image_square = np.pad(cropped_image, ((0, 0), (pad_before, pad_after), (0, 0)))
    else:
        cropped_image_square = np.pad(cropped_image, ((pad_before, pad_after), (0, 0), (0, 0)))
        
    cropped_image_square_shape = cropped_image_square.shape
    cropped_sqaure_header = rescaled_header
    cropped_sqaure_header['dim'][1:4] = cropped_image_square_shape
    cropped_sqaure_affine = rescaled_affine
    cropped_sqaure_affine = nib.affines.rescale_affine(rescaled_affine, cropped_image_square_shape, (1,1,1))
                        
    cropped_image_square_nifti = nib.Nifti1Image(cropped_image_square,
                                                  header = cropped_sqaure_header,
                                                  affine = cropped_sqaure_affine)
    file_name_3D = path.split('\\')[-1]
    os.makedirs(FolderPath_3D, exist_ok=True)
    nib.save(cropped_image_square_nifti, os.path.join(FolderPath_3D, file_name_3D))
    
    cropped_sqaure_header_2D = cropped_sqaure_header
    cropped_sqaure_header_2D['dim'][3] = 1
    cropped_image_square_shape_2D = cropped_sqaure_header_2D['dim'][1:4]
    cropped_sqaure_affine_2D = cropped_sqaure_affine
    cropped_sqaure_affine_2D = nib.affines.rescale_affine(cropped_sqaure_affine, cropped_image_square_shape_2D, (1,1,1))
    for index in range(0, cropped_image_square_shape[-1]):
        image_temp = cropped_image_square[:, :, index:index+1]
        image_temp_nifti = nib.Nifti1Image(image_temp,
                                            header = cropped_sqaure_header_2D,
                                            affine = cropped_sqaure_affine_2D)
        file_name_2D = str(index) + '.nii.gz'
        os.makedirs(FolderPath_2D, exist_ok=True)
        nib.save(image_temp_nifti, os.path.join(FolderPath_2D, file_name_2D))
    
    if plot_debug:
        # image_numpy = sitk_to_numpy(cropped_image)
        image_numpy = cropped_image
        
        plt.imshow(image_numpy[:, :, 0], cmap='bone')
        plt.title('Cropped End Diastolic Image')
        plt.axis('off')
        plt.show()
        plt.close()
        
        plt.imshow(image_numpy[:, :, 10], cmap='bone')
        plt.title('Cropped End Systolic Image')
        plt.axis('off')
        plt.show()
        plt.close()
        

if __name__ == '__main__':
    # Input directory
    patient_dir = "Parent_Directory"
    
    # Output directory & folders (2D & 3D)
    out_patient_dir = "Destination_Directory"
    folder_2D = '2D_saveout_folder_name'
    folder_3D = '3D_saveout_folder_name'

    for patient in os.listdir(patient_dir):
        patient_path = os.path.join(patient_dir, patient)
        patient_folder_path = os.path.join(patient_dir, patient, 'case_folder_name')
        for data in os.listdir(patient_folder_path):
            image_path = os.path.join(patient_folder_path, data)
            FolderPath_3D = os.path.join(out_patient_dir, patient, folder_3D)
            FolderPath_2D = os.path.join(out_patient_dir, patient, folder_2D)
            print(patient)
            detect_roi(image_path, FolderPath_3D, FolderPath_2D, True)
# Adaptive-Heart-Extraction

This algoritihm is designed to automatically identify the cardiac structure in all Cine MRI frames, based on the assumption that the heart is the largest image structure that moves across the cardiac cycle. 

Expected Data Type: NIFTI
Expected Data Dimension: 3D (2D + Time)
Expected Folder Structure as followed:
          Parent Directory (the input path)
          - Case Folder
            - Case File

After the user downloaded all the scripts to a local python environment, only need to change the input and destination folder names in run_detection_github.py to run the algorithm.

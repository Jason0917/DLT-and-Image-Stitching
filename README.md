# DLT for homography estimation, Image Stitching, Vanishing Point Detection
## Perspective Correction
Please use the following command to run PerspectiveCorrection.py

```python PerspectiveCorrection.py -i images_folder_path -n npy_path```

If you want to use normalized DLT, run

```python PerspectiveCorrection.py -i images_folder_path -n npy_path --norm```

Example

```python PerspectiveCorrection.py -i dlt/images -n dlt/gt --norm```

## Feature Extraction and Matching & Image Stitching

Please use the following command to run ImageStitching.py

```python ImageStitching.py -i images_folder_path```

Close the pop-out image window to continue running the program.

If you want to use LMEDS instead of RANSAC, please run

```python ImageStitching.py -i images_folder_path --lmeds```

Example 

```python ImageStitching.py -i overdetermined/fishbowl --lmeds```

## Vanishing Point Detection

Please use the following command to run Vanishing_pt.py

```python Vanishing_pt.py -i image_path```

Example 

```python Vanishing_pt.py -i vanishing_pt/carla.png```

import os

model_path = 'C:/Users/ASUS/OneDrive/Desktop/opencv/res10_300x300_ssd_iter_140000.caffemodel'
if os.path.exists(model_path) and os.access(model_path, os.R_OK):
    print("File exists and is readable")
else:
    print("File does not exist or is not readable")
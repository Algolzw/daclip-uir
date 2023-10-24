import numpy as np
import cv2
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.deg_util import degrade


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# deg_type: noisy, jpeg
# param: 50 for noise_level, 10 for jpeg compression quality
def generate_LQ(deg_type='blur', param=50):
    print(deg_type, param)
    # set data dir
    sourcedir = "datasets/universal/val/noisy/GT"
    savedir = "datasets/universal/val/noisy/LQ"

    if not os.path.isdir(sourcedir):
        print("Error: No source data found")
        exit(0)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    filepaths = [f for f in os.listdir(sourcedir) if is_image_file(f)]
    num_files = len(filepaths)

    # prepare data with augementation
    for i in range(num_files):
        filename = filepaths[i]
        print("No.{} -- Processing {}".format(i, filename))
        # read image
        image = cv2.imread(os.path.join(sourcedir, filename)) / 255.

        image_LQ = (degrade(image, deg_type, param) * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(savedir, filename), image_LQ)
        
    print('Finished!!!')

if __name__ == "__main__":
    generate_LQ()

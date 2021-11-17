import cv2
import os
import time

dirname = os.path.dirname(__file__)
inference_outdir = os.path.join(dirname,"inference/output")

for file in sorted(os.listdir(inference_outdir)):
    print(file)
    image = cv2.imread(os.path.join(inference_outdir, file))
    cv2.imshow("image", image)
    cv2.waitKey(0)
    time.sleep(1/25)
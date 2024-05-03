import os
import cv2

for file in os.listdir("../clouds_data/train/cloud/"):
    image = cv2.imread(f"../clouds_data/train/cloud/{file}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"../clouds_data/train/cloud/{file}", gray)

for file in os.listdir("../clouds_data/train/not_cloud/"):
    image = cv2.imread(f"../clouds_data/train/not_cloud/{file}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"../clouds_data/train/not_cloud/{file}", gray)

for file in os.listdir("../clouds_data/test/cloud/"):
    image = cv2.imread(f"../clouds_data/test/cloud/{file}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"../clouds_data/test/cloud/{file}", gray)

for file in os.listdir("../clouds_data/test/not_cloud/"):
    image = cv2.imread(f"../clouds_data/test/not_cloud/{file}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"../clouds_data/test/not_cloud/{file}", gray)

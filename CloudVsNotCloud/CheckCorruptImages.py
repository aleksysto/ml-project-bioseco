from struct import unpack
from tqdm import tqdm
import os


marker_mapping = {
    0xffd8: "Start of Image",
    0xffe0: "Application Default Header",
    0xffdb: "Quantization Table",
    0xffc0: "Start of Frame",
    0xffc4: "Define Huffman Table",
    0xffda: "Start of Scan",
    0xffd9: "End of Image"
}


class JPEG:
    def __init__(self, image_file):
        with open(image_file, 'rb') as f:
            self.img_data = f.read()

    def decode(self):
        data = self.img_data
        while (True):
            marker, = unpack(">H", data[0:2])
            # print(marker_mapping.get(marker))
            if marker == 0xffd8:
                data = data[2:]
            elif marker == 0xffd9:
                return
            elif marker == 0xffda:
                data = data[-2:]
            else:
                lenchunk, = unpack(">H", data[2:4])
                data = data[2+lenchunk:]
            if len(data) == 0:
                break


bads = []
images = ["../clouds_data/test/not_cloud/", "../clouds_data/test/cloud/", "../clouds_data/train/cloud/", "../clouds_data/train/not_cloud/"]

for path in images:
    for img in tqdm(os.listdir(path)):
        image = f"{path}/{img}"
        image = JPEG(image)
        try:
            image.decode()
        except:
            bads.append(f"{path}{img}")


for name in bads:
    print(name)

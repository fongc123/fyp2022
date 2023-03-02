import utils
import time
import cv2
import os
import json

SRC = "./data_orange_peels/"
DST = "./data_orange_peels_split/"
NSPLITS = 8

def main():
    replications = {}

    try: os.stat(DST)
    except FileNotFoundError: os.mkdir(DST)

    for subdir in os.listdir(SRC):
        print("Processing class", subdir, "...")
        try: os.stat(os.path.join(DST, subdir))
        except FileNotFoundError: os.mkdir(os.path.join(DST, subdir))

        idx = 0
        for file in os.listdir(os.path.join(SRC, subdir)):
            if not file.startswith("IMG"):
                continue
            
            image = cv2.cvtColor(cv2.imread(os.path.join(SRC, subdir, file)), cv2.COLOR_BGR2RGB)
            cnt = utils.find_object_boundary_canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), dilate_size=3)
            masked = utils.draw_masked_image(image, cnt)
            images = utils.filter_subimages(utils.split_image(masked, nsplits=NSPLITS))
            for subimage in images:
                cv2.imwrite(os.path.join(DST, subdir, f"{idx}.png"), cv2.cvtColor(subimage, cv2.COLOR_RGB2BGR))
                idx += 1
            replications[file] = len(images)

        print("Processed", idx, "images.")

    with open(os.path.join(DST, "replications.json"), "w") as f:
        json.dump(replications, f)

if __name__ == "__main__":
    print("Splitting images...")
    start = time.time()
    main()
    print("Time Elapsed:", time.time() - start)
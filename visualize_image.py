import cv2
import numpy as np
import os
import json
import skimage
from pyzbar.pyzbar import decode

from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib

from mrcnn.visualize import apply_mask

ROOT_DIR = r"C:\Users\Aysenur\Desktop\Project"
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

MY_MODEL_PATH = os.path.join(ROOT_DIR, "my_mask_rcnn_weights.h5")


class CustomConfig(Config):
    """Configuration for training on the custom dataset.
    """
    # Give the configuration name
    NAME = "object"

    GPU_COUNT = 1

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 4  # Background + Pudding, Milk, Juice, Pasta

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 145

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.90


config = CustomConfig()
config.display()


class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):

        self.add_class("object", 1, "Pudding")
        self.add_class("object", 2, "Milk")
        self.add_class("object", 3, "Pasta")
        self.add_class("object", 4, "Juice")

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        annotations1 = json.load(
            open("annotations.json"))

        annotations = list(annotations1.values())  # don't need the dict keys

        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:

            polygons = [r['shape_attributes'] for r in a['regions']]
            objects = [s['region_attributes']['names'] for s in a['regions']]

            name_dict = {"Pudding": 1, "Milk": 2, "Pasta": 3, "Juice": 4}
            num_ids = [name_dict[a] for a in objects]

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
            )

    def load_mask(self, image_id):

        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):

            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

            mask[rr, cc, i] = 1

        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids 

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

model.load_weights(MY_MODEL_PATH, by_name=True)

class_names = ["BG", "Pudding", "Milk", "Pasta", "Juice"]


def random_colors(N):
    colors = [tuple(np.random.rand(3)) for _ in range(N)]
    return colors


colors = random_colors(len(class_names))
class_dict = {
    name: color for name, color in zip(class_names, colors)
}


def find_barcode(image):
    num_barcode = []  
    if decode(image):
        for barcode in decode(image):
            pts = np.array([barcode.polygon], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, (0, 255, 0), 5)

            num_barcode.append([barcode.data])

    return len(num_barcode)


def display_instances(image, frame_name, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]        

    for i in range(n_instances):

        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox.
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        color = class_dict[label]
        score = round(scores[i]*100, 2) if scores is not None else None

        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]
        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 0, 0), 1)
        image = cv2.putText(image, f'Number of objects: {n_instances}', \
            (0, 25), cv2.FONT_HERSHEY_PLAIN, 1, (100, 0, 255), 1)
    num_barcode = find_barcode(image)
    # cv2.putText(image, f'Number of barcodes: {num_barcode}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (100, 0, 255), 1)    

    return image

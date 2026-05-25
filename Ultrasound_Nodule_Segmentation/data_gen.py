import os
import cv2
import numpy as np
from tensorflow import keras

class DataGen(keras.utils.Sequence):
    def __init__(self, ids, base_path, classes, batch_size=8, image_size=128):
        self.ids = ids
        self.base_path = base_path
        self.classes = classes
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()

    def __load__(self, id_name):
        for class_name in self.classes:
            image_path = os.path.join(self.base_path, class_name, "images", id_name) + ".png"
            mask_path = os.path.join(self.base_path, class_name, "masks/")
            
            # Check if the image file exists
            if not os.path.exists(image_path):
                continue

            print(f"Trying to read image from: {image_path}")  # Debugging output
            image = cv2.imread(image_path, 1)
            if image is None:
                print(f"Failed to read image: {image_path}")
                continue

            image = cv2.resize(image, (self.image_size, self.image_size))
            
            mask = np.zeros((self.image_size, self.image_size, 1))
            
            all_masks = os.listdir(mask_path)
            for name in all_masks:
                if id_name not in name:  # Ensure the mask corresponds to the image
                    continue
                _mask_path = os.path.join(mask_path, name)
                print(f"Trying to read mask from: {_mask_path}")  # Debugging output

                _mask_image = cv2.imread(_mask_path, -1)
                if _mask_image is None:
                    print(f"Failed to read mask: {_mask_path}")
                    continue
                _mask_image = cv2.resize(_mask_image, (self.image_size, self.image_size))
                _mask_image = np.expand_dims(_mask_image, axis=-1)
                mask = np.maximum(mask, _mask_image)
                
            image = image / 255.0
            mask = mask / 255.0

            return image, mask

        print(f"Image ID not found in any class: {id_name}")
        return None, None

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.ids):
            files_batch = self.ids[index * self.batch_size:]
        else:
            files_batch = self.ids[index * self.batch_size : (index + 1) * self.batch_size]
        
        images = []
        masks  = []
        
        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            if _img is not None and _mask is not None:
                images.append(_img)
                masks.append(_mask)
            
        images = np.array(images)
        masks  = np.array(masks)
        
        return images, masks

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))

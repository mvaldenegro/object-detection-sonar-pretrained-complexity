import os
import random
from xml.etree import ElementTree
import numpy as np

label_map = {
    0:  'Background',
    1:  'Bottle',
    2:  'Can',
    3:  'Chain',
    4:  'Drink-carton',
    5:  'Hook',
    6:  'Propeller',
    7:  'Shampoo-bottle',
    8:  'Standing-bottle',
    9:  'Tire',
    10: 'Valve',
    11: 'Wall',
}

class Marine:
    def __init__(self, path, class_names, split='train'):
        self.path = path
        self.arg_to_class = class_names
        self.class_to_arg = {value: key for key, value
                             in class_names.items()}
        self.class_names = list(class_names.values())
        self.num_classes = len(class_names)
        image_files, annotation_files = self.get_filenames(path)
        (self.train_images, self.train_annotations, self.val_images,
         self.val_annotations, self.test_images, self.test_annotations
         ) = self.split_files(image_files, annotation_files)
        self.split = split
        if split == 'train':
            # Save the test image and use it for trained model
            np.savetxt('test_set.txt', np.array(self.test_images), fmt='%s')

    def get_filenames(self, path):
        images_path = os.path.join(path, 'Images')
        image_names = os.listdir(images_path)
        random.shuffle(image_names)
        image_files = []
        annotation_files = []
        for files in image_names:
            absolute_file = os.path.join(images_path, files)
            if absolute_file:
                image_files.append(absolute_file)
            path = absolute_file.replace('/Images/', '/Annotations/')
            filename = path.split('.png')[0] + '.xml'
            if os.path.exists(filename):
                annotation_files.append(filename)
        return image_files, annotation_files

    def split_files(self, image_files, annotation_files):
        total_images = len(image_files)
        train_split = int(0.7 * total_images)
        val_split = int(0.2 * total_images)
        train_annotations = annotation_files[:train_split]
        train_images = image_files[:train_split]
        val_annotations = annotation_files[train_split:
                                           train_split + val_split]
        val_images = image_files[train_split: train_split + val_split]
        test_annotations = annotation_files[train_split + val_split:]
        test_images = image_files[train_split + val_split:]
        return (train_images, train_annotations, val_images, val_annotations,
                test_images, test_annotations)

    def get_split_data(self, split_name):
        if split_name == 'train':
            return self.train_images, self.train_annotations
        elif split_name == 'val':
            return self.val_images, self.val_annotations
        elif split_name == 'test':
            images = np.loadtxt('test_set.txt', dtype='str')
            replacer = lambda x: x.replace('/Images/', '/Annotations/')
            ext_changer = lambda x: x.replace('.png', '.xml')
            replacer = np.vectorize(replacer)
            ext_changer = np.vectorize(ext_changer)
            annotations = replacer(images)
            annotations = ext_changer(annotations)
            return images, annotations
        else:
            raise ValueError('Give proper split name.')

    def load_data(self, split_name):
        images, annotations = self.get_split_data(split_name)
        data = []
        for idx in range(len(images)):
            tree = ElementTree.parse(annotations[idx])
            root = tree.getroot()
            size_tree = root.find('size')
            width = float(size_tree.find('width').text)
            height = float(size_tree.find('height').text)
            if split_name == 'test':
                width, height = 1, 1
            box_data = []
            for object_tree in root.findall('object'):
                class_name = object_tree.find('name').text
                if class_name in self.class_to_arg:
                    class_arg = self.class_to_arg[class_name]
                    bounding_box = object_tree.find('bndbox')

                    x = float(bounding_box.find('x').text)
                    y = float(bounding_box.find('y').text)
                    w = float(bounding_box.find('w').text)
                    h = float(bounding_box.find('h').text)
                    xmin = x
                    ymin = y
                    xmax = xmin + w
                    ymax = ymin + h
                    xmin, xmax = xmin / width, xmax / width
                    ymin, ymax = ymin / height, ymax / height

                    box_data.append([xmin, ymin, xmax, ymax, class_arg])

            data.append({'image': images[idx], 'boxes': np.asarray(box_data),
                         'image_index': images[idx]})

        return data

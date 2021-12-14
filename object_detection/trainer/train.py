import os
import argparse

import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from paz.optimization.callbacks import LearningRateScheduler
from paz.models import SSD300
from paz.optimization import MultiBoxLoss
from paz.abstract import ProcessingSequence

from paz.processors import TRAIN, VAL
from object_detection.dataloader.marine_debris import Marine
from object_detection.dataloader.marine_debris import label_map
from object_detection.trainer.pipelines import DetectSingleShotGray
from object_detection.trainer.pipelines import AugmentDetection
from object_detection.trainer.pipelines import EvaluateMAPGray


description = 'Training script for single-shot object detection models'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-dsp', '--dataset_path',
                    default='data/watertank-segmentation/', type=str,
                    help='Marine debris dataset path')
parser.add_argument('-bs', '--batch_size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('-et', '--evaluation_period', default=3, type=int,
                    help='evaluation frequency')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                    help='Initial learning rate for SGD')
parser.add_argument('-m', '--momentum', default=0.9, type=float,
                    help='Momentum for SGD')
parser.add_argument('-g', '--gamma_decay', default=0.1, type=float,
                    help='Gamma decay for learning rate scheduler')
parser.add_argument('-e', '--num_epochs', default=240, type=int,
                    help='Maximum number of epochs before finishing')
parser.add_argument('-iou', '--AP_IOU', default=0.5, type=float,
                    help='Average precision IOU used for evaluation')
parser.add_argument('-sp', '--save_path', default='trained_models/',
                    type=str, help='Path for writing model weights and logs')
parser.add_argument('-dp', '--data_path', default='VOCdevkit/',
                    type=str, help='Path for writing model weights and logs')
parser.add_argument('-se', '--scheduled_epochs', nargs='+', type=int,
                    default=[110, 152], help='Epoch learning rate reduction')
parser.add_argument('-mp', '--multiprocessing', default=False, type=bool,
                    help='Select True for multiprocessing')
parser.add_argument('-w', '--workers', default=1, type=int,
                    help='Number of workers used for optimization')
args = parser.parse_args()

optimizer = Adam(args.learning_rate, args.momentum)

Dataset = Marine(args.dataset_path, label_map, 'train')
train_data = Dataset.load_data('train')
val_data = Dataset.load_data('val')
test_data = Dataset.load_data('test')

train_data = train_data[:5]
val_data = val_data[:5]
test_data = test_data[:5]

datasets = list()
datasets.append(train_data)
datasets.append(val_data)
class_names = Dataset.class_names
num_classes = Dataset.num_classes
input_image_shape = (480, 320)

model = SSD300(num_classes, base_weights=None, head_weights=None)
model.summary()

# Instantiating loss and metrics
loss = MultiBoxLoss()
metrics = {'boxes': [loss.localization,
                     loss.positive_classification,
                     loss.negative_classification]}
model.compile(optimizer, loss.compute_loss, metrics)

# setting data augmentation pipeline
augmentators = []
for split in [TRAIN, VAL]:
    augmentator = AugmentDetection(model.prior_boxes, split,
                                   num_classes=num_classes)
    augmentators.append(augmentator)

# setting sequencers
sequencers = []
for data, augmentator in zip(datasets, augmentators):
    sequencer = ProcessingSequence(augmentator, args.batch_size, data)
    sequencers.append(sequencer)

# batch = sequencers[0].__getitem__(0)
# print(batch[0]['image'].shape)

# setting callbacks
model_path = os.path.join(args.save_path, model.name)
if not os.path.exists(model_path):
    os.makedirs(model_path)
log = CSVLogger(os.path.join(model_path, model.name + '-optimization.log'))
save_path = os.path.join(model_path, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
checkpoint = ModelCheckpoint(save_path, verbose=1, save_weights_only=True)
schedule = LearningRateScheduler(
    args.learning_rate, args.gamma_decay, args.scheduled_epochs)
evaluate = EvaluateMAPGray(
    test_data,
    DetectSingleShotGray(model, Dataset.class_names, 0.01, 0.45),
    args.evaluation_period,
    args.save_path,
    args.AP_IOU, class_names)

# training
model.fit(
    sequencers[0],
    epochs=args.num_epochs,
    verbose=1,
    callbacks=[checkpoint, log, schedule, evaluate],
    validation_data=sequencers[1],
    use_multiprocessing=args.multiprocessing,
    workers=0)

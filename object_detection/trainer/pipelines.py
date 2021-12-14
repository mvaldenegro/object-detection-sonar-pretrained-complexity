import os
import numpy as np
from paz import processors as pr
from paz.abstract import SequentialProcessor, Processor
from paz.pipelines.image import AugmentImage
from tensorflow.keras.callbacks import Callback
from paz.evaluation import evaluateMAP


class StackImages(Processor):
    """Converts grayscale image to 3 channel input for SSD.

    # Arguments
        image: Image of shape (height, width)
    """

    def __init__(self):
        super(StackImages, self).__init__()

    def call(self, image):
        image = np.stack((image,) * 3, axis=-1)
        return image


class AugmentBoxes(SequentialProcessor):
    """Perform data augmentation with bounding boxes.
    # Arguments
        mean: List of three elements used to fill empty image spaces.
    """
    def __init__(self, mean=pr.BGR_IMAGENET_MEAN):
        super(AugmentBoxes, self).__init__()
        self.add(pr.ToImageBoxCoordinates())
        self.add(pr.Expand(mean=mean))
        self.add(pr.RandomSampleCrop())
        self.add(pr.RandomFlipBoxesLeftRight())
        self.add(pr.ToNormalizedBoxCoordinates())


class PreprocessBoxes(SequentialProcessor):
    """Preprocess bounding boxes
    # Arguments
        num_classes: Int.
        prior_boxes: Numpy array of shape ``[num_boxes, 4]`` containing
            prior/default bounding boxes.
        IOU: Float. Intersection over union used to match boxes.
        variances: List of two floats indicating variances to be encoded
            for encoding bounding boxes.
    """
    def __init__(self, num_classes, prior_boxes, IOU, variances):
        super(PreprocessBoxes, self).__init__()
        self.add(pr.MatchBoxes(prior_boxes, IOU),)
        self.add(pr.EncodeBoxes(prior_boxes, variances))
        self.add(pr.BoxClassToOneHotVector(num_classes))


class PreprocessImage(SequentialProcessor):
    """Preprocess RGB image by resizing it to the given ``shape``. If a
    ``mean`` is given it is substracted from image and it not the image gets
    normalized.
    # Arguments
        shape: List of two Ints.
        mean: List of three Ints indicating the per-channel mean to be
            subtracted.
    """
    def __init__(self, shape, mean=None):
        super(PreprocessImage, self).__init__()
        self.add(pr.ResizeImage(shape))
        self.add(pr.CastImage(float))
        self.add(pr.NormalizeImage())


class AugmentDetection(SequentialProcessor):
    """Augment boxes and images for object detection.
    # Arguments
        prior_boxes: Numpy array of shape ``[num_boxes, 4]`` containing
            prior/default bounding boxes.
        split: Flag from `paz.processors.TRAIN`, ``paz.processors.VAL``
            or ``paz.processors.TEST``. Certain transformations would take
            place depending on the flag.
        num_classes: Int.
        size: Int. Image size.
        mean: List of three elements indicating the per channel mean.
        IOU: Float. Intersection over union used to match boxes.
        variances: List of two floats indicating variances to be encoded
            for encoding bounding boxes.
    """
    def __init__(self, prior_boxes, split=pr.TRAIN, num_classes=11, size=300,
                 mean=None, IOU=.5, variances=[0.1, 0.1, 0.2, 0.2]):
        super(AugmentDetection, self).__init__()
        # image processors
        self.augment_image = AugmentImage()
        self.preprocess_image = PreprocessImage((size, size), mean)

        # box processors
        self.augment_boxes = AugmentBoxes()
        args = (num_classes, prior_boxes, IOU, variances)
        self.preprocess_boxes = PreprocessBoxes(*args)

        # pipeline
        self.add(pr.UnpackDictionary(['image', 'boxes']))
        self.add(pr.ControlMap(pr.LoadImage(), [0], [0]))
        if split == pr.TRAIN:
            self.add(pr.ControlMap(self.augment_image, [0], [0]))
            self.add(pr.ControlMap(self.augment_boxes, [0, 1], [0, 1]))
        self.add(pr.ControlMap(self.preprocess_image, [0], [0]))
        self.add(pr.ControlMap(self.preprocess_boxes, [1], [1]))
        self.add(pr.SequenceWrapper(
            {0: {'image': [size, size, 3]}},
            {1: {'boxes': [len(prior_boxes), 4 + num_classes]}}))


class DetectSingleShotGray(Processor):
    """Single-shot object detection prediction.
    # Arguments
        model: Keras model.
        class_names: List of strings indicating the class names.
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        mean: List of three elements indicating the per channel mean.
        draw: Boolean. If ``True`` prediction are drawn in the returned image.
    """
    def __init__(self, model, class_names, score_thresh, nms_thresh,
                 mean=None, variances=[0.1, 0.1, 0.2, 0.2],
                 draw=True):
        self.model = model
        self.class_names = class_names
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.variances = variances
        self.draw = draw

        super(DetectSingleShotGray, self).__init__()
        preprocessing = SequentialProcessor(
            [pr.ResizeImage(self.model.input_shape[1:3]),
             pr.NormalizeImage(),
             pr.CastImage(float),
             pr.ExpandDims(axis=0)])
        postprocessing = SequentialProcessor(
            [pr.Squeeze(axis=None),
             pr.DecodeBoxes(self.model.prior_boxes, self.variances),
             pr.NonMaximumSuppressionPerClass(self.nms_thresh),
             pr.FilterBoxes(self.class_names, self.score_thresh)])
        self.predict = pr.Predict(self.model, preprocessing, postprocessing)

        self.denormalize = pr.DenormalizeBoxes2D()
        self.draw_boxes2D = pr.DrawBoxes2D(self.class_names)
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])

    def call(self, image):
        boxes2D = self.predict(image)
        boxes2D = self.denormalize(image, boxes2D)
        if self.draw:
            image = self.draw_boxes2D(image, boxes2D)
        return self.wrap(image, boxes2D)


class EvaluateMAPGray(Callback):
    """Evaluates mean average precision (MAP) of an object detector.
    # Arguments
        data_manager: Data manager and loader class. See ''paz.datasets''
            for examples.
        detector: Tensorflow-Keras model.
        period: Int. Indicates how often the evaluation is performed.
        save_path: Str.
        iou_thresh: Float.
    """
    def __init__(
            self, dataset, detector, period, save_path, iou_thresh=0.5,
            class_names=None):
        super(EvaluateMAPGray, self).__init__()
        self.detector = detector
        self.period = period
        self.save_path = save_path
        self.dataset = dataset
        self.iou_thresh = iou_thresh
        self.class_names = class_names
        self.class_dict = {}
        for class_arg, class_name in enumerate(self.class_names):
            self.class_dict[class_name] = class_arg

    def on_epoch_end(self, epoch, logs):
        if (epoch + 1) % self.period == 0:
            result = evaluateMAP(
                self.detector,
                self.dataset,
                self.class_dict,
                iou_thresh=self.iou_thresh,
                use_07_metric=True)

            result_str = 'mAP: {:.4f}\n'.format(result['map'])
            metrics = {'mAP': result['map']}
            for arg, ap in enumerate(result['ap']):
                if arg == 0 or np.isnan(ap):  # skip background
                    continue
                metrics[self.class_names[arg]] = ap
                result_str += '{:<16}: {:.4f}\n'.format(
                    self.class_names[arg], ap)
            print(result_str)

            # Saving the evaluation results
            filename = os.path.join(self.save_path, 'MAP_Evaluation_Log.txt')
            with open(filename, 'a') as eval_log_file:
                eval_log_file.write('Epoch: {}\n{}\n'.format(
                    str(epoch), result_str))

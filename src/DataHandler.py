import tensorflow as tf
import os
import pickle

def get_parse_fn(mode, params):
    """
    Creates a parse function for dataset examples. Performs decoding, reshaping and casting on both train and eval
    data, performs preprocessing on train data if turned on.
    """
    def _parse_fn(example):
        """
        Method to parse dataset content. Decodes images, reshapes and casts them to float32, applies preprocessing
        when necessary.
        """

        """
        Reads dataset information from pickle file. Contents:
            'encoded':              True if dataset has been encoded, false if tf.decode_raw() is fine
            'data':                 Name of image data feature in format dict
            'labels':               Name of label data feature in format dict
            'resize_before_use':    True if images need to be resized/normalised before use
            'hwd':                  Array containing height, width and depth of encoded image data
            'eval_batch_size':      Batch size used for evaluation, either whole dataset or large batches
            'file_pattern':         Dictionary containing the file patterns for tfrecord files ('train', 'validation', 'test')
            'format':               Dictionary containing format of dataset to decode
        """
        with open(os.path.join(params.data_dir, 'content.pickle'), 'rb') as handle:
            info = pickle.load(handle)

        example_fmt = info['format']
        parsed = tf.parse_single_example(example, example_fmt)

        # decode data, dependant on encoding
        if info['encoded']:
            image = tf.image.decode_image(parsed[info['data']], channels=3, dtype=tf.float32)
        else:
            image = tf.cast(tf.decode_raw(parsed[info['data']], tf.uint8), tf.float32)

        # reshape data into desired shape
        if info['resize_before_use']:
            # used when dataset has varying image sizes
            # first, grab actual image size from parsed Example content
            height = parsed['image/height']
            width  = parsed['image/width']
            depth  = parsed['image/channels']
            # set image to actual shape first
            image = tf.reshape(image, tf.stack([height, width, depth]))
            # find target shape next, resize into that with cropping or 0-padding
            height, width, depth = info['hwd']
            # resizes image to size that fits in `size` to preserve aspect ratio
            image = tf.image.resize_images(image, [height, width], preserve_aspect_ratio=True)
            # resize again without preservation to get uniform image size with minimal distortion
            image = tf.image.resize_images(image, [height, width], preserve_aspect_ratio=False)
        else:
            # no preprocessing necessary, set to static image size
            height, width, depth = info['hwd']
            image.set_shape([height * width * depth])
            image = tf.reshape(image, [height, width, depth])
        label = parsed[info['labels']]

        # perform preprocessing if enabled
        if mode == 'train' and params.preprocess_data:
            zoom_factor = params.preprocess_zoom
            image = tf.image.resize_image_with_crop_or_pad(image, int(height*zoom_factor), int(width*zoom_factor))
            image = tf.random_crop(image, [height, width, depth])
            image = tf.image.random_flip_left_right(image)
        return image, label

    return _parse_fn

class DataHandler():
    """Class to provide datasets from serialised .tfrecords-files."""
    def __init__(self, mode, file_pattern, params, parse_fn=get_parse_fn):
        with open(os.path.join(params.data_dir, 'content.pickle'), 'rb') as handle:
            info = pickle.load(handle)

        files = tf.data.Dataset.list_files(os.path.join(params.data_dir, info['file_pattern'][file_pattern]))
        self.dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=1) # cycle_length = num files
        self.params = params
        self.parse_fn = parse_fn
        self.mode = mode

    def prepare_for_train(self):
        """Performs mapping on dataset, forms batches, shuffles and repeats and performs prefetching."""
        self.dataset = self.dataset.apply(tf.contrib.data.map_and_batch(
            map_func=self.parse_fn(self.mode, self.params),
            batch_size=self.params.batch_size,
            num_parallel_batches=self.params.num_cores,
            drop_remainder=True))
        self.dataset = self.dataset.apply(tf.contrib.data.shuffle_and_repeat(
            buffer_size=self.params.batch_size,
            count=None)) # repeat forever, assumes estimator stops after max_steps
        self.dataset = self.dataset.prefetch(buffer_size=1)
        return self.dataset

    def prepare_for_eval(self, eval_batch_size=-1):
        """Performs mapping on dataset, forms batches, no shuffling or repeating, no prefetching."""
        with open(os.path.join(self.params.data_dir, 'content.pickle'), 'rb') as handle:
            info = pickle.load(handle)
        
        self.dataset = self.dataset.apply(tf.contrib.data.map_and_batch(
            map_func=self.parse_fn(self.mode, self.params),
            batch_size=(info['eval_batch_size'] if eval_batch_size == -1 else eval_batch_size),
            num_parallel_batches=self.params.num_cores,
            drop_remainder=True))
        return self.dataset
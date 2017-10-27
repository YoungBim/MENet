import os
import numpy as np
import tensorflow as tf
from scipy import misc

def preprocess(image, batch_size, height, width, annotation=None):
    '''
    Performs preprocessing for one set of image and annotation for feeding into network.
    NO scaling of any sort will be done as per original paper.
    INPUTS:
    - image (Tensor): the image input 3D Tensor of shape [height, width, 3]
    - annotation (Tensor): the annotation input 3D Tensor of shape [height, width, 1]
    - height (int): the output height to reshape the image and annotation into
    - width (int): the output width to reshape the image and annotation into
    OUTPUTS:
    - preprocessed_image(Tensor): the reshaped image tensor
    - preprocessed_annotation(Tensor): the reshaped annotation tensor
    '''

    #Convert the image and annotation dtypes to tf.float32 if needed
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # image = tf.cast(image, tf.float32)

    image = tf.image.resize_image_with_crop_or_pad(image, height, width)
    image.set_shape(shape=(batch_size, height, width, 3))

    if not annotation == None:
        annotation = tf.image.resize_image_with_crop_or_pad(annotation, height, width)
        annotation.set_shape(shape=(batch_size, height, width, 1))

        return image, annotation
    return image

def datasetAsList(dataset_dir, TaskDirs, Tasks):
    '''
        Function dedicated to parse the dataset and return a list of files
    '''
    image_files = {}
    annotation_files = {}

    for task in Tasks:  # For each task of the network
        # Seek for the full list of raw images
        dataset_raw_path = os.path.join(dataset_dir, TaskDirs[task] + "_train_raw")
        dataset_gt_path = os.path.join(dataset_dir, TaskDirs[task] + "_train_gt")
        pngfiles = np.array([os.path.join(root, name).replace('\\','/')
                             for root, _, files in os.walk(dataset_raw_path, followlinks=True)
                             for name in files
                             if name.endswith(".png")])

        # Check the existence of the GT file (i.e. is that an unsupervised sample or a supervised one ?!)
        Supervised = np.array(
            [os.path.isfile(os.path.join(dataset_gt_path, filename.split('/')[-2], filename.split('/')[-1]).replace('\\','/'))
             for filename in pngfiles
             ])

        # Remove all samples that are unsupervised
        image_files[task] = pngfiles[Supervised]

        # Reorder the files by name (just for style)
        image_files[task] = sorted(image_files[task])

        # Generate the set of annotation path accordingly
        annotation_files[task] = np.array([
            os.path.join(dataset_gt_path, item.split('/')[-2], item.split('/')[-1]).replace('\\','/')
            for item in image_files[task]
        ])
    return image_files, annotation_files
#
def bytes_feature(value):
    '''
    Creates a TensorFlow Record Feature with value as a byte array
    '''
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def int64_feature(value):
    '''
    Creates a TensorFlow Record Feature with value as a 64 bit integer.
    '''
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float_feature(value):
    '''
    Creates a TensorFlow Record Feature with value as a float.
    '''
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def write_record(dest_path, df):
    '''
    Writes an actual TF record from a data frame
    '''
    writer = tf.python_io.TFRecordWriter(dest_path)
    for i in range(len(df['image'])):
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': int64_feature(df['height'][i]),
            'width': int64_feature(df['width'][i]),
            'depth': int64_feature(df['depth'][i]),
            'image': bytes_feature(df['image'][i]),
            'annotation': bytes_feature(df['annotation'][i]),
            'task': int64_feature(df['task'][i])
        }))
        writer.write(example.SerializeToString())
    writer.close()

def read_image_to_bytestring(path):
    '''
    Reads an image from a path and converts it
    to a flattened byte string
    '''
    img = misc.imread(path)
    if (img.dtype == np.int32):
        img = ((img * 255)/65535).astype(np.uint8)
    return img.flatten().tostring()

def write_records_from_file(image_files, annotation_files, taskid, taskname, dest_folder, num_records):
    '''
    Takes a label file as a path and converts entries into a tf record
    for image classification.
    '''
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)


    start_idx = 0
    ex_per_rec = np.uint(np.ceil(len(image_files) / num_records))
    for i in range(1, num_records):
        rec_path = dest_folder + taskname + '_' + str(i) + '.tfrecords'
        # read image, flatten and then convert to a string
        _image_files_ = image_files[start_idx:(ex_per_rec * i)]
        _annotation_files_ = annotation_files[start_idx:(ex_per_rec * i)]
        img_arrs = {}
        img_arrs['height'] = [misc.imread(path).shape[0] for path in _image_files_]
        img_arrs['width'] = [misc.imread(path).shape[1] for path in _image_files_]
        img_arrs['depth'] = [misc.imread(path).shape[2] for path in _image_files_]
        img_arrs['image'] = [read_image_to_bytestring(path) for path in _image_files_]
        img_arrs['annotation'] = [read_image_to_bytestring(path) for path in _annotation_files_]
        img_arrs['task'] = [np.int64(taskid) for _ in range(len(_image_files_))]
        write_record(rec_path, img_arrs)
        start_idx += ex_per_rec
        print('wrote record: ', i)

    # The versy last tf.reccord might be a bit smaller....
    final_rec_path = dest_folder + taskname + '_' + str(num_records) + '.tfrecords'
    _image_files_ = image_files[start_idx:]
    _annotation_files_ = annotation_files[start_idx:]
    img_arrs = {}
    img_arrs['height'] = [misc.imread(path).shape[0] for path in _image_files_]
    img_arrs['width'] = [misc.imread(path).shape[1] for path in _image_files_]
    img_arrs['depth'] = [misc.imread(path).shape[2] for path in _image_files_]
    img_arrs['image'] = [read_image_to_bytestring(path) for path in _image_files_]
    img_arrs['annotation'] = [read_image_to_bytestring(path) for path in _annotation_files_]
    img_arrs['task'] = [np.int64(taskid) for _ in range(len(_image_files_))]
    write_record(final_rec_path, img_arrs)
    print('wrote record: ', num_records)
    print('finished writing ' + taskname + ' records...')

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#
def _parse_features(parsed_features, batch_size = None ):
    '''
    Parse each element of the dataset and transfo
    '''

    if batch_size is None:
        print('XXXXXXXXXXXXXXXXXXXXXXX')
    # Parse the features
    height = tf.cast(parsed_features['height'],tf.int32)
    height.set_shape(shape=(1))
    width = tf.cast(parsed_features['width'],tf.int32)
    width.set_shape(shape=(1))
    depth = tf.cast(parsed_features['depth'],tf.int32)
    depth.set_shape(shape=(1))
    image_shape = tf.stack([batch_size, height[0], width[0], depth[0]], name='img_shape')
    annots_shape = tf.stack([batch_size, height[0], width[0], 1], name='annots_shape')

    image = tf.decode_raw(parsed_features['image'], tf.uint8, name='decode_image')
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, image_shape, name='reshape_image')

    annotation = tf.decode_raw(parsed_features['annotation'], tf.uint8, name='decode_annotation')
    annotation = tf.reshape(annotation, annots_shape, name='reshape_annotation')

    task = tf.cast(parsed_features['task'], tf.uint8)
    task.set_shape(shape=(batch_size))

    image, annotation = preprocess(image=image, batch_size=batch_size, width=360, height=480, annotation=annotation)

    result = {'image': image, 'annotation': annotation, 'task': task}
    return result


def _parse_function(example_proto, batch_size = None):
    '''
    Function dedicated parse the features from the tf.reccord
    '''
    features = {
                'image': tf.FixedLenFeature([], tf.string, default_value=""),
                'annotation': tf.FixedLenFeature([], tf.string, default_value=""),
                'task': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
                'height': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
                'width': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
                'depth': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
    }

    #parsed_features = tf.parse_single_example(example_proto, features)
    parsed_features = tf.parse_example(example_proto, features)
    processsed_features = _parse_features(parsed_features, batch_size=batch_size)
    return processsed_features
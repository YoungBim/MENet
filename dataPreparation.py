import os
import numpy as np
import tensorflow as tf
from PIL import Image

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
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

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
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

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
def bytes_feature(value):
    '''
    Creates a TensorFlow Record Feature with value as a byte array
    '''
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
def int64_feature(value):
    '''
    Creates a TensorFlow Record Feature with value as a 64 bit integer.
    '''
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
def float_feature(value):
    '''
    Creates a TensorFlow Record Feature with value as a float.
    '''
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
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

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
def read_image_to_bytestring(path):
    '''
    Reads an image from a path and converts it
    to a flattened byte string
    '''
    img = np.asarray(Image.open(path))
    shape = img.shape
    if (img.dtype == np.int32):
        img = ((img * 255)/65535)
    img = img.astype(np.uint8)
    return img.flatten().tostring(), shape

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
def write_records_from_file(image_files, annotation_files, taskid, taskname, dest_folder, num_records):
    '''
    Takes a label file as a path and converts entries into a tf record
    for image classification.
    '''
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    def filelistToimg_arrs(_image_files_, _annotation_files_):
        img_arrs = {}
        temp_reccord = [read_image_to_bytestring(path) for path in _image_files_]
        img_arrs['image'] = [elmt[0] for elmt in temp_reccord]
        img_arrs['height'] = [elmt[1][0] for elmt in temp_reccord]
        img_arrs['width'] = [elmt[1][1] for elmt in temp_reccord]
        img_arrs['depth'] = [elmt[1][2] for elmt in temp_reccord]
        img_arrs['annotation'] = [read_image_to_bytestring(path)[0] for path in _annotation_files_]
        img_arrs['task'] = [np.int64(taskid) for _ in range(len(_image_files_))]
        return img_arrs

    start_idx = 0
    ex_per_rec = np.uint(np.ceil(len(image_files) / num_records))
    for i in range(1, num_records):
        rec_path = dest_folder + taskname + '_' + str(i) + '.tfrecords'
        # read image, flatten and then convert to a string
        _image_files_ = image_files[int(start_idx):int(ex_per_rec * i)]
        _annotation_files_ = annotation_files[int(start_idx):int(ex_per_rec * i)]
        img_arrs = filelistToimg_arrs(_image_files_, _annotation_files_)
        write_record(rec_path, img_arrs)
        start_idx += ex_per_rec
        print('wrote record: ', i)

    # The versy last tf.reccord might be a bit smaller....
    final_rec_path = dest_folder + taskname + '_' + str(num_records) + '.tfrecords'
    _image_files_ = image_files[int(start_idx):]
    _annotation_files_ = annotation_files[int(start_idx):]
    img_arrs = filelistToimg_arrs(_image_files_, _annotation_files_)
    write_record(final_rec_path, img_arrs)
    print('wrote record: ', num_records)
    print('finished writing ' + taskname + ' records...')

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
def _parse_features(parsed_features, batch_size = None ):
    '''
    Parse each element of the dataset and transfo
    '''

    if batch_size is None:
        print('XXXXXXXXXXXXXXXXXXXXXXX')
    # Parse the features
    height = tf.cast(parsed_features['height'],tf.int32)
    height.set_shape(shape=(batch_size))
    width = tf.cast(parsed_features['width'],tf.int32)
    width.set_shape(shape=(batch_size))
    depth = tf.cast(parsed_features['depth'],tf.int32)
    depth.set_shape(shape=(batch_size))


    for batch_item in range(batch_size):
        image_shape = tf.stack([1, height[batch_item], width[batch_item], depth[batch_item]], name='img_shape')
        annots_shape = tf.stack([1, height[batch_item], width[batch_item], 1], name='annots_shape')

        img = tf.decode_raw(parsed_features['image'][batch_item], tf.uint8, name='decode_image')
        img = tf.cast(img, tf.float32)
        img = tf.reshape(img, image_shape, name='reshape_image')
        # Convert img from BGR to RGB and revert scale 0-255
        ant = tf.decode_raw(parsed_features['annotation'][batch_item], tf.uint8, name='decode_annotation')
        ant = tf.reshape(ant, annots_shape, name='reshape_annotation')

        task_btch = tf.cast(parsed_features['task'][batch_item], tf.uint8)
        task_btch.set_shape(shape=())

        image_btch, annotation_btch = preprocess(image=img, batch_size=1, width=480, height=360, annotation=ant)
        if (batch_item > 0):
            image = tf.concat([image, image_btch], axis = 0)
            annotation = tf.concat([annotation, annotation_btch], axis = 0)
            task = tf.concat([task, tf.reshape(task_btch,[1])], axis = 0)
        else:
            image = image_btch
            annotation = annotation_btch
            task = tf.reshape(task_btch,[1])
    result = {'image': image, 'annotation': annotation, 'task': task}
    return result

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
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
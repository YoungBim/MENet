import os
import numpy as np
import tensorflow as tf
from scipy import misc

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
            'task': int64_feature(df['task'][i]),
            'image': bytes_feature(df['image'][i]),
            'annotation': bytes_feature(df['annotation'][i])
        }))
        writer.write(example.SerializeToString())
    writer.close()

def read_image_to_bytestring(path):
    '''
    Reads an image from a path and converts it
    to a flattened byte string
    '''
    img = misc.imread(path)
    img_shape = img.shape
    return img.reshape(img_shape).flatten().tostring()

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
    img_arrs['image'] = [read_image_to_bytestring(path) for path in _image_files_]
    img_arrs['annotation'] = [read_image_to_bytestring(path) for path in _annotation_files_]
    img_arrs['task'] = [np.int64(taskid) for _ in range(len(_image_files_))]
    write_record(final_rec_path, img_arrs)
    print('wrote record: ', num_records)
    print('finished writing ' + taskname + 'records...')


if __name__ == '__main__':
    print('toto')
import os
import tensorflow as tf
import numpy as np
from itertools import chain
from preprocessing import preprocess

dataset_dir = "C:\DL\MENet\dataset_win"

Tasks = ["segmentation", "depth"]
TaskDirs = {Tasks[0]: 'seg', Tasks[1]: 'depth'}
TaskLabel = {Tasks[i]: np.uint8(i) for i in range(len(Tasks))}
image_height = 360
image_width = 480

# ===============PREPARATION FOR TRAINING==================
image_files = {}
annotation_files = {}

for task in Tasks:  # For each task of the network
    # Seek for the full list of raw images
    dataset_raw_path = os.path.join(dataset_dir, TaskDirs[task] + "_train_raw")
    dataset_gt_path = os.path.join(dataset_dir, TaskDirs[task] + "_train_gt")
    pngfiles = np.array([os.path.join(root, name).replace('\\', '/')
                         for root, _, files in os.walk(dataset_raw_path, followlinks=True)
                         for name in files
                         if name.endswith(".png")])

    # Check the existence of the GT file (i.e. is that an unsupervised sample or a supervised one ?!)
    Supervised = np.array(
        [os.path.isfile(
            os.path.join(dataset_gt_path, filename.split('/')[-2], filename.split('/')[-1]).replace('\\', '/'))
         for filename in pngfiles
         ])

    # Remove all samples that are unsupervised
    image_files[task] = pngfiles[Supervised]

    # Generate the set of annotation path accordingly
    annotation_files[task] = np.array([
        os.path.join(dataset_gt_path, item.split('/')[-2], item.split('/')[-1]).replace('\\', '/')
        for item in image_files[task]
    ])

    image_files[task] = sorted(image_files[task])
    annotation_files[task] = sorted(annotation_files[task])

# Reorder the files by name (just for style)
for task in Tasks:
    image_files[task] = sorted(image_files[task])
    annotation_files[task] = sorted(annotation_files[task])

print('Loaded')

def load_OneImage():

        images = tf.placeholder(dtype=tf.string,shape=[1],name='ImagePaths')
        annotations = tf.placeholder(dtype=tf.string, shape=[1], name='AnotPath')
        tasks = tf.placeholder(dtype=tf.uint8, shape=[1], name='Task')

        # Decode the image and annotation raw content
        filename = tf.identity(images[0])
        image = tf.read_file(images[0])
        image = tf.image.decode_image(image, channels=3)
        annotation = tf.read_file(annotations[0])
        tsk = tasks[0]

        annotation = tf.image.decode_image(annotation, channels=None, name='decode_images')
        annotation = tf.reduce_mean(annotation, axis=2)
        annotation = tf.expand_dims(annotation, -1)

        # preprocess and batch up the image & annotation
        img, ants = preprocess(image, annotation, image_height, image_width)

        return filename, img, ants, tsk, images, annotations, tasks

# Define the graph
batch_filename, batch_img, batch_annots, batch_task,  ph_images,  ph_annotations,  ph_tasks = load_OneImage()

# Define the session superviser
sv = tf.train.Supervisor(logdir='./log/',
                         save_summaries_secs=0,
                         saver=None)


# Actually runs the session
with sv.managed_session() as sess:

    # Load the files into one input queue
    imlist = np.array(list(chain.from_iterable([image_files[task] for task in Tasks])))
    anotlist = np.array(list(chain.from_iterable([annotation_files[task] for task in Tasks])))
    taskslist = np.array(
        list(chain.from_iterable([np.tile(TaskLabel[task], len(annotation_files[task])) for task in Tasks])),
        dtype=np.uint8)


    fetches = {}
    fetches["filename"] = batch_filename
    fetches["img"] = batch_img
    fetches["annots"] = batch_annots
    fetches["task"] = batch_task
    for idx in range(len(imlist)):
        if taskslist[idx:idx+1] == 0:
            tabs = ":\t"
        else:
            tabs = ":\t\t\t"
        print('Start', idx, ' \t', Tasks[taskslist[idx]],tabs, imlist[idx])
        feeddict = {ph_images : imlist[idx:idx+1], ph_annotations : anotlist[idx:idx+1], ph_tasks : taskslist[idx:idx+1] }
        results = sess.run(fetches,feeddict)
        if results["task"] == 0:
            tabs = ":\t"
        else:
            tabs = ":\t\t\t"
        print('Ending \t\t', Tasks[results["task"]],tabs, results["filename"])

# Found So far : hamburg_000000_054029
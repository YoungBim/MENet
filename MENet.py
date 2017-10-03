from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
from PIL import Image
slim = tf.contrib.slim

from enet import ENetEncoder, ENetSegDecoder, ENetDepthDecoder, ENet_arg_scope
from get_class_weights import ENet_weighing, median_frequency_balancing
from preprocessing import preprocess
from itertools import compress, chain
from losses import depth_loss_nL1, segmentation_loss_wce


class MENet(object):


    def __init__(self, FLAGS, Tasks, TaskDirs, TaskLabel):
        self.Tasks = Tasks
        self.TaskDirs = TaskDirs
        self.TaskLabel = TaskLabel
        # ==========NAME HANDLING FOR CONVENIENCE==============
        self.opt = FLAGS
        assert (len(self.TaskDirs.values()) == len(self.Tasks))

    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    # Function dedicated to the data preparation (i.e. pure python preprocessing)
    def prepare_Data(self):
        # ===============PREPARATION FOR TRAINING==================
        image_files = {}
        annotation_files = {}
        SubDataSets = {}
        Supervised = {}

        for task in self.Tasks:  # For each task of the network
            # Seek for the full list of raw images
            dataset_raw_path = os.path.join(self.opt.dataset_dir, self.TaskDirs[task] + "_train_raw")
            dataset_gt_path = os.path.join(self.opt.dataset_dir, self.TaskDirs[task] + "_train_gt")
            pngfiles = np.array([os.path.join(root, name)
                                 for root, _, files in os.walk(dataset_raw_path)
                                 for name in files
                                 if name.endswith(".png")])

            # Get the list of the sub datasets per task
            subdataset = np.array([pngfiles[i].split('/')[-2] for i in range(len(pngfiles))])
            SubDataSets[task] = subdataset[np.insert(subdataset[:-1] != subdataset[1:], 0, True)].tolist()

            # Check the existence of the GT file (i.e. is that an unsupervised sample or a supervised one ?!)
            Supervised[task] = np.array(
                [os.path.isfile(os.path.join(dataset_gt_path, filename.split('/')[-2], filename.split('/')[-1]))
                 for filename in pngfiles
                 ])

            # Remove all samples that are unsupervised
            image_files[task] = pngfiles[Supervised[task]]

            # Generate the set of annotation path accordingly
            annotation_files[task] = np.array([
                os.path.join(dataset_gt_path, item.split('/')[-2], item.split('/')[-1])
                for item in image_files[task]
            ])

        # Know the number steps to take before decaying the learning rate and batches per epoch
        num_batches_per_epoch = 0
        for task in self.Tasks:
            num_batches_per_epoch = num_batches_per_epoch + len(image_files[task])
        self.opt.num_batches_per_epoch = num_batches_per_epoch / self.opt.batch_size
        self.opt.num_steps_per_epoch = num_batches_per_epoch
        self.opt.decay_steps = int(self.opt.num_epochs_before_decay * self.opt.num_steps_per_epoch)

        # =================CLASS WEIGHTS===============================
        # Median frequency balancing class_weights
        if self.opt.weighting == "MFB":
            self.class_weights = median_frequency_balancing(annotation_files["segmentation"])

        # Inverse weighing probability class weights
        elif self.opt.weighting == "ENET":
            self.class_weights = ENet_weighing(annotation_files["segmentation"])

        return image_files, annotation_files

    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    # Function dedicated to queue data loading (batches)
    def load_Data(self, image_files, annotation_files):
            # Load the files into one input queue
            imlist = np.array(list(chain.from_iterable([image_files[task] for task in self.Tasks])))
            images = tf.convert_to_tensor(imlist)
            anotlist = np.array(list(chain.from_iterable([annotation_files[task] for task in self.Tasks])))
            annotations = tf.convert_to_tensor(anotlist)
            taskslist = np.array(
                list(chain.from_iterable([np.tile(self.TaskLabel[task], len(annotation_files[task])) for task in self.Tasks])),
                dtype=np.uint8)
            tasks = tf.convert_to_tensor(taskslist, dtype=tf.uint8)

            input_queue = tf.train.slice_input_producer(
                [images, annotations, tasks])  # Slice_input producer shuffles the data by default.

            # Decode the image and annotation raw content
            image = tf.read_file(input_queue[0])
            image = tf.image.decode_image(image, channels=3)
            annotation = tf.read_file(input_queue[1])
            task = input_queue[2]

            annotation = tf.image.decode_image(annotation, channels=None, name='decode_images')
            annotation = tf.reduce_mean(annotation, axis=2)
            annotation = tf.expand_dims(annotation, -1)

            # preprocess and batch up the image & annotation
            preprocessed_image, preprocessed_annotation = preprocess(image, annotation, self.opt.image_height, self.opt.image_width)

            images, annotations, tasks = tf.train.batch([preprocessed_image, preprocessed_annotation, task],
                                                        batch_size=self.opt.batch_size, allow_smaller_final_batch=True)
            return images, annotations, tasks


    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    # Function dedicated to compute the inference from the model given the images
    def MENet_Model(self, images):
        with slim.arg_scope(ENet_arg_scope(weight_decay=self.opt.weight_decay)):
            # Define the shared encoder
            inputs_shapes = ENetEncoder(    images,
                                            batch_size=self.opt.batch_size,
                                            is_training=True,
                                            reuse=None,
                                            num_initial_blocks=self.opt.num_initial_blocks,
                                            stage_two_repeat=self.opt.stage_two_repeat,
                                            skip_connections=self.opt.skip_connections)

            # Collect tensors that are useful later (e.g. tf summary)
            self.predictions = {}

            # Define the decoder(s)
            for task in self.Tasks:
                if (task == 'segmentation'):
                    logits, probabilities = ENetSegDecoder(  inputs_shapes,
                                                             self.opt.num_classes,
                                                             is_training=True,
                                                             reuse=None,
                                                             stage_two_repeat=self.opt.stage_two_repeat,
                                                             skip_connections=self.opt.skip_connections)
                    self.probabilities = probabilities
                    self.predictions[task] = tf.identity(logits, name=task + '_pred')

                elif (task == 'depth'):
                    disparity = ENetDepthDecoder(   inputs_shapes,
                                                    is_training=True,
                                                    reuse=None)
                    self.predictions[task] = tf.identity(disparity, name=task + '_pred')


    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    # Function dedicated to compute the task specific losses
    def compute_loss(self, task, pred, anots, n_smpl):
        if task == "segmentation":
            loss = tf.cond(tf.greater(n_smpl, tf.constant(0, dtype=tf.float32)),
                           lambda: segmentation_loss_wce(task, pred, anots, self.opt.num_classes, self.class_weights),
                           lambda: tf.constant(0, dtype=tf.float32))
        elif task == "depth":
            loss = tf.cond(tf.greater(n_smpl, tf.constant(0, dtype=tf.float32)),
                           lambda: depth_loss_nL1(task, pred, anots), lambda: tf.constant(0, dtype=tf.float32))
        else:
            print("Please define a loss for this task")
            loss = tf.constant(0, dtype=tf.float32)
        return loss

    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    # Function dedicated to compute the loss from the inference predictions
    def MENet_Loss(self, annotations, tasks):
        loss = {}  # Task dependent loss
        n_smpl = {}  # Task dependent number of samples in batch
        has_smpl = {}  # Storage of the fact that the current batch has samples for the given task
        mask = {}  # Task dependent batch mask
        anots = {}  # Task dependent annotations
        pred = {}  # Task dependent predictions
        for task in self.Tasks:
            # Create a mask to filter the batch depending on the task
            mask[task] = tf.equal(tasks, self.TaskLabel[task], name=task + '_mask')
            # Count the number of task-related samples
            n_smpl[task] = tf.reduce_sum(tf.cast(mask[task], tf.float32), name=task + '_n_smpl')
            # Define if the task has a sample in the batch
            has_smpl[task] = tf.greater(n_smpl[task], 0.0)
            # Filter the task dedicated annotations
            anots[task] = tf.squeeze(tf.boolean_mask(annotations, mask[task]), axis=3, name=task + '_anot_mask')
            # Filter the task dedicated predictions
            pred[task] = tf.boolean_mask(self.predictions[task], mask[task], name=task + '_pred_mask')
            # Compute the loss associated to each task
            loss[task] = tf.identity(
                tf.cond(has_smpl[task], lambda: self.compute_loss(task, pred[task], anots[task], n_smpl[task]),
                        lambda: tf.constant(0, dtype=tf.float32)), name=task + '_loss')
            loss[task] = tf.cond(has_smpl[task], lambda: tf.multiply(loss[task],n_smpl[task]), lambda : tf.constant([0.0],dtype=tf.float32))
            loss[task] = tf.identity(tf.divide(loss[task],self.opt.batch_size), name = 'norm_loss_' + task)
        # Collect tensors that are useful later (e.g. tf summary)
        self.mask = mask
        self.n_smpl = n_smpl
        self.has_smpl = has_smpl
        self.pred = pred
        self.anots = anots
        self.losses = loss
        losses = [tf.squeeze(loss[task]) for task in self.Tasks]
        self.total_loss = tf.add_n(losses,name='total_loss')

    def Optimize(self):
        train_vars = [var for var in tf.trainable_variables()]

        self.global_step = tf.Variable(0,
                                       name='global_step',
                                       trainable=False)
        self.incr_global_step = tf.assign(self.global_step,
                                          self.global_step + 1)

        # Define your exponentially decaying learning rate
        self.opt.learning_rate = tf.train.exponential_decay(
            learning_rate=self.opt.initial_learning_rate,
            global_step=self.global_step,
            decay_steps=self.opt.decay_steps,
            decay_rate=self.opt.learning_rate_decay_factor,
            staircase=False)

        optim = tf.train.AdamOptimizer(self.opt.learning_rate, self.opt.adam_momentum)
        self.train_op = slim.learning.create_train_op(self.total_loss,optim)


    def build_train_graph(self):
        opt = self.opt
        with tf.name_scope("Data"):
            image_files, annotation_files = self.prepare_Data()
            images, annotations, tasks = self.load_Data(image_files, annotation_files)

        with tf.name_scope("Model"):
            self.MENet_Model(images)

        with tf.name_scope("Loss"):
            self.MENet_Loss(annotations, tasks)

        with tf.name_scope("Optimizer"):
            self.Optimize()


    def collect_summaries(self):
        with tf.name_scope("SummaryGeneration"):
            # Now finally create all the summaries you need to monitor and group them into one summary op.
            for task in self.Tasks:
                tf.summary.scalar('Monitor/' + task + '_Loss', self.losses[task])
                tf.summary.scalar('Monitor/' + task + '_Samples', self.n_smpl[task])

            tf.summary.scalar('Monitor/Total_Loss', self.total_loss)
            tf.summary.scalar('Monitor/learning_rate', self.opt.learning_rate)

            self.images2write_gt = {}
            self.images2write = {}
            for task in self.Tasks:
                if task == 'segmentation':
                    # Create an output for showing the segmentation output of validation images
                    prob = tf.boolean_mask(self.probabilities, self.mask[task])
                    segmentation_pred_val = tf.reshape(tf.cast(tf.argmax(prob, axis=-1), dtype=tf.float32),
                                                       shape=[-1, self.opt.image_height, self.opt.image_width, 1])
                    segmentation_pred_val = tf.expand_dims(segmentation_pred_val[0, :, :, :], axis=0)
                    tf.summary.image('Images/pred_' + task, segmentation_pred_val, max_outputs=1)
                    segmentation_gt_val = tf.cast(
                        tf.reshape(tf.cast(self.anots[task], dtype=tf.float32), shape=[-1, self.opt.image_height, self.opt.image_width, 1]),
                        dtype=tf.float32)
                    segmentation_gt_val = tf.expand_dims(segmentation_gt_val[0,:,:,:],axis=0)
                    tf.summary.image('Images/gt_' + task, segmentation_gt_val, max_outputs=1)

                    # Save the images to be written later
                    self.images2write[task] = tf.squeeze(segmentation_pred_val,axis = [0, 3])
                    self.images2write_gt[task] = tf.squeeze(segmentation_gt_val,axis = [0, 3])

                else:
                    disp_pred = tf.expand_dims(self.pred[task][0,:,:,:], axis=0) # Make sure the items are synced
                    tf.summary.image('Images/pred' + task, disp_pred, max_outputs=1)
                    depth_gt_val = tf.expand_dims(tf.expand_dims(self.anots[task],axis=-1)[0,:,:,:], axis=0) # Make sure the items are synced
                    tf.summary.image('Images/gt_' + task, depth_gt_val, max_outputs=1)

                    # Save the images to be written later
                    self.images2write[task] = tf.squeeze(disp_pred,axis = [0, 3])
                    self.images2write_gt[task] = tf.squeeze(depth_gt_val,axis = [0, 3])

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
        if step == 'latest':
            self.saver.save(sess,
                            os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            self.saver.save(sess,
                            os.path.join(checkpoint_dir, model_name),
                            global_step=step)

    def train(self):

        self.build_train_graph()
        self.collect_summaries()

        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                            for v in tf.trainable_variables()])
            self.saver = tf.train.Saver([var for var in tf.trainable_variables()] + \
                                    [self.global_step],
                                    save_relative_paths=True,
                                    max_to_keep=self.opt.max_model_saved)
        sv = tf.train.Supervisor(logdir=self.opt.logdir,
                                 save_summaries_secs=0,
                                 saver=None)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with sv.managed_session(config=config) as sess:
            print('Trainable variables: ')
            for var in tf.trainable_variables():
                print(var.name)
            print("parameter_count =", sess.run(parameter_count))

            for step in xrange(int(self.opt.num_steps_per_epoch * self.opt.num_epochs)):
                start_time = time.time()
                fetches = {
                    "train": self.train_op,
                    "global_step": self.global_step,
                    "incr_global_step": self.incr_global_step
                }

                if step % self.opt.summary_freq == 0:
                    fetches["loss"] = self.total_loss
                    fetches["losses"] = self.losses
                    fetches["summary"] = sv.summary_op

                if self.opt.save_images and step % self.opt.save_model_freq == 0:
                    fetches["images2write"] = self.images2write
                    fetches["images2write_gt"] = self.images2write_gt

                results = sess.run(fetches)
                gs = results["global_step"]

                if step % self.opt.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil(gs / self.opt.num_batches_per_epoch)
                    train_step = gs - (train_epoch - 1) * self.opt.num_batches_per_epoch
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it" \
                            % (train_epoch, train_step, self.opt.num_batches_per_epoch, \
                                time.time() - start_time))

                    pt = "\t losses : "
                    for task in self.Tasks:
                        pt = pt + task + " : " + str(results["losses"][task]) + " | "
                    print(pt + "total : %.3f"%(results["loss"]))

                if step % self.opt.save_model_freq == 0:
                    self.save(sess, self.opt.logdir, 'latest')
                    self.save(sess, self.opt.logdir, gs)
                    if self.opt.save_images:
                        # Write images prediction vs GT
                        im2write = results["images2write"]
                        im2write_gt = results["images2write_gt"]
                        for task in self.Tasks:
                            # The predicted images must be converted
                            img_tens = im2write[task]
                            if (img_tens.ptp()>0):
                                if task == 'depth':
                                    img_tens = 255.0 * (img_tens - img_tens.min()) / (img_tens.ptp())
                                elif task == 'segmentation':
                                    img_tens = 255.0 * img_tens / (self.opt.num_classes-1)
                            img_tens = np.uint8(img_tens)
                            img = Image.fromarray(img_tens)
                            img.save(os.path.join(self.opt.logdir, task + "_" + str(gs) + "_pred.jpeg"))
                            # Write the GT depth
                            img_tens = im2write_gt[task]
                            if task == 'segmentation':
                                img_tens = 255.0 * img_tens / (self.opt.num_classes-1)
                            img_tens = img_tens.astype(np.uint8)
                            img = Image.fromarray(img_tens)
                            img.save(os.path.join(self.opt.logdir, task + "_" + str(gs) + "_gt.jpeg"))
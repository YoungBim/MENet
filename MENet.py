from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from PIL import Image
slim = tf.contrib.slim

from enet import ENetEncoder, ENetSegDecoder, ENetDepthDecoder, ENet_arg_scope
from get_class_weights import ENet_weighing, median_frequency_balancing
from preprocessing import preprocess
from itertools import chain
from losses import depth_loss_nL1_Reg, segmentation_loss_wce_Reg


class MENet(object):


    def __init__(self, FLAGS, Tasks, TaskDirs, TaskLabel):
        self.Tasks = Tasks
        self.TaskDirs = TaskDirs
        self.TaskLabel = TaskLabel
        self.opt = FLAGS
        assert (len(self.TaskDirs.values()) == len(self.Tasks))
        self.opt.ImagesDirectory = os.path.join(self.opt.logdir,'Images/')
        self.SessionConfig = tf.ConfigProto()
        self.SessionConfig.gpu_options.allow_growth = True

    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    # Function dedicated to the data preparation (i.e. pure python preprocessing)
    def prepare_Data(self):
        # ===============PREPARATION FOR TRAINING==================
        image_files = {}
        annotation_files = {}

        for task in self.Tasks:  # For each task of the network
            # Seek for the full list of raw images
            dataset_raw_path = os.path.join(self.opt.dataset_dir, self.TaskDirs[task] + "_train_raw")
            dataset_gt_path = os.path.join(self.opt.dataset_dir, self.TaskDirs[task] + "_train_gt")
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

            # Generate the set of annotation path accordingly
            annotation_files[task] = np.array([
                os.path.join(dataset_gt_path, item.split('/')[-2], item.split('/')[-1]).replace('\\','/')
                for item in image_files[task]
            ])

            image_files[task] = sorted(image_files[task])
            annotation_files[task] = sorted(annotation_files[task])

        # Reorder the files by name (just for style)
        for task in self.Tasks:
            image_files[task] = sorted(image_files[task])
            annotation_files[task] = sorted(annotation_files[task])

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

            self.batch_images, self.batch_annotations, self.batch_tasks = tf.train.batch([preprocessed_image, preprocessed_annotation, task],
                                                        batch_size=self.opt.batch_size, allow_smaller_final_batch=True)


    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    # Function dedicated to compute the inference from the model given the images
    def MENet_Model(self):
        with slim.arg_scope(ENet_arg_scope(weight_decay=self.opt.weight_decay)):
            # Define the shared encoder
            Encoder = ENetEncoder(          self.batch_images,
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
                    logits, probabilities = ENetSegDecoder(  Encoder,
                                                             self.opt.num_classes,
                                                             is_training=True,
                                                             reuse=None,
                                                             stage_two_repeat=self.opt.stage_two_repeat,
                                                             skip_connections=self.opt.skip_connections)
                    self.probabilities = probabilities
                    self.predictions[task] = tf.identity(logits, name=task + '_pred')

                elif (task == 'depth'):
                    disparity = ENetDepthDecoder(   Encoder,
                                                    skip_connections=self.opt.skip_connections,
                                                    is_training=True,
                                                    reuse=None)
                    self.predictions[task] = tf.identity(disparity, name=task + '_pred')


    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    # Function dedicated to compute the task specific losses
    def compute_loss(self, task, pred, anots, n_smpl):
        if task == "segmentation":
            loss = tf.cond(tf.greater(n_smpl, tf.constant(0, dtype=tf.float32)),
                           lambda: segmentation_loss_wce_Reg(task, pred, anots, self.opt.num_classes, self.class_weights),
                           lambda: tf.constant(0, dtype=tf.float32))
        elif task == "depth":
            loss = tf.cond(tf.greater(n_smpl, tf.constant(0, dtype=tf.float32)),
                           lambda: depth_loss_nL1_Reg(task, pred, anots), lambda: tf.constant(0, dtype=tf.float32))
        else:
            print("Please define a loss for this task")
            loss = tf.constant(0, dtype=tf.float32)
        return loss

    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    # Function dedicated to compute the loss from the inference predictions
    def MENet_Loss(self):
        loss = {}  # Task dependent loss
        n_smpl = {}  # Task dependent number of samples in batch
        has_smpl = {}  # Storage of the fact that the current batch has samples for the given task
        mask = {}  # Task dependent batch mask
        anots = {}  # Task dependent annotations
        pred = {}  # Task dependent predictions
        for task in self.Tasks:
            # Create a mask to filter the batch depending on the task
            mask[task] = tf.equal(self.batch_tasks, self.TaskLabel[task], name=task + '_mask')
            # Count the number of task-related samples
            n_smpl[task] = tf.reduce_sum(tf.cast(mask[task], tf.float32), name=task + '_n_smpl')
            # Define if the task has a sample in the batch
            has_smpl[task] = tf.greater(n_smpl[task], 0.0)
            # Filter the task dedicated annotations
            anots[task] = tf.squeeze(tf.boolean_mask(self.batch_annotations, mask[task]), axis=3, name=task + '_anot_mask')
            # Filter the task dedicated predictions
            pred[task] = tf.boolean_mask(self.predictions[task], mask[task], name=task + '_pred_mask')
            # Compute the loss associated to each task
            loss[task] = tf.cond(has_smpl[task], lambda: self.compute_loss(task, pred[task], anots[task], n_smpl[task]),
                        lambda: tf.constant(0, dtype=tf.float32), name=task + '_loss')
            #loss[task] = tf.multiply(loss[task],n_smpl[task])
            #loss[task] = tf.identity(tf.divide(loss[task],self.opt.batch_size), name = 'norm_loss_' + task)
        # Collect tensors that are useful later (e.g. tf summary)
        self.mask = mask
        self.n_smpl = n_smpl
        self.has_smpl = has_smpl
        self.pred = pred
        self.anots = anots
        self.losses = loss
        losses = [tf.squeeze(loss[task]) for task in self.Tasks]
        self.total_loss = tf.add_n(losses,name='total_loss')

    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    # Function dedicated to compute the loss from the inference predictions
    def Optimize(self):
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
            staircase=True)

        optim = tf.train.AdamOptimizer(self.opt.learning_rate, self.opt.adam_momentum)
        self.train_op = slim.learning.create_train_op(self.total_loss,optim)

    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    # Function dedicated to compute the loss from the inference predictions
    def build_train_graph(self):

        image_files, annotation_files = self.prepare_Data()

        with tf.name_scope("Data"):
            self.load_Data(image_files, annotation_files)

        with tf.name_scope("Model"):
            self.MENet_Model()

        with tf.name_scope("Loss"):
            self.MENet_Loss()

        with tf.name_scope("Optimizer"):
            self.Optimize()

    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    # Function dedicated to drop the task-dependent summaries
    def TaskDependent_Summary(self,task):

        # Display the loss in the summary
        tf.summary.scalar(task + '/Loss', self.losses[task])
        # Display the #of smples / batch in the summary
        tf.summary.scalar(task + '/Samples', self.n_smpl[task])


        # Task-Dependent summaries
        if task == 'segmentation':
            def getSegmentation_pred():
                return tf.expand_dims(self.pred[task][0, :, :, :], axis=0)
            def getNoSegmentation_pred():
                return tf.zeros([1, self.opt.image_height, self.opt.image_width, self.opt.num_classes], dtype=tf.float32, name="No_Segmentation")
            # Put a prediction from the batch to the summary (if exists in the batch)
            img2sum = tf.cond(self.has_smpl[task], getSegmentation_pred, getNoSegmentation_pred)
            img2sum = tf.reshape(tf.cast(tf.argmax(img2sum, axis=-1), dtype=tf.float32),
                                 shape=[-1, self.opt.image_height, self.opt.image_width, 1])
            tf.summary.image(task + '/pred', img2sum, max_outputs=1)
            # Save the images to be written later
            self.images2write[task + '_pred'] = tf.squeeze(img2sum, axis=[0, 3])

            def getSegmentation_gt():
                return tf.expand_dims(self.anots[task][0, :, :], axis=0)
            def getNoSegmentation_gt():
                return tf.zeros([1, self.opt.image_height, self.opt.image_width], dtype=tf.uint8, name="No_Segmentation_gt")
            # Put a gt from the batch to the summary (if exists in the batch)
            img2sum = tf.cond(self.has_smpl[task], getSegmentation_gt, getNoSegmentation_gt)
            img2sum = tf.cast(
                tf.reshape(tf.cast(img2sum, dtype=tf.float32),
                           shape=[-1, self.opt.image_height, self.opt.image_width, 1]),
                dtype=tf.float32)
            tf.summary.image(task + '/gt', img2sum, max_outputs=1)
            # Save the images to be written later
            self.images2write[task + '_gt'] = tf.squeeze(img2sum, axis=[0, 3])

            for othertask in self.Tasks:
                if not othertask == task:
                    def getSegmentation_depth_pred():
                        temp = tf.boolean_mask(self.predictions[othertask], self.mask[task])
                        return tf.expand_dims(temp[0, :, :, :], axis=0)
                    def getNoSegmentation_depth_pred():
                        return tf.zeros([1, self.opt.image_height, self.opt.image_width, 1], dtype=tf.float32, name="No_Segmentation_depth_pred")
                    # Put a prediction OF THE OTHER TASK from the batch to the summary
                    img2sum = tf.cond(self.has_smpl[task], getSegmentation_depth_pred, getNoSegmentation_depth_pred)
                    # Put a prediction OF THE OTHER TASK from the batch to the summary (if exists in the batch)
                    tf.summary.image(task + '/pred_' + othertask, img2sum, max_outputs=1)
                    # Save the images to be written later
                    self.images2write[task + '_pred_' + othertask] = tf.squeeze(img2sum, axis=[0, 3])

        elif task == 'depth':
            def getDepth_pred():
                return tf.expand_dims(self.pred[task][0, :, :, :], axis=0)
            def getNoDepth_pred():
                return tf.zeros([1, self.opt.image_height, self.opt.image_width, 1], dtype=tf.float32, name="No_Depth")
            # Put a prediction image from the batch to the summary  (if exists in the batch)
            img2sum = tf.cond(self.has_smpl[task], getDepth_pred, getNoDepth_pred)
            tf.summary.image(task + '/pred', img2sum, max_outputs=1)
            # Save the images to be written later
            self.images2write[task + '_pred']  = tf.squeeze(img2sum,axis = [0, 3])

            def getDepth_gt():
                return tf.expand_dims(tf.expand_dims(self.anots[task][0,:,:],axis=-1), axis=0)
            def getNoDepth_gt():
                return tf.zeros([1, self.opt.image_height, self.opt.image_width, 1], dtype=tf.uint8, name="No_Depth_gt")
            # Put a gt image from the batch to the summary  (if exists in the batch)
            img2sum = tf.cond(self.has_smpl[task], getDepth_gt, getNoDepth_gt)
            tf.summary.image(task + '/gt', img2sum, max_outputs=1)
            # Save the images to be written later
            self.images2write[task + '_gt'] = tf.squeeze(img2sum,axis = [0, 3])

            for othertask in self.Tasks:
                if not othertask == task:
                    def getDepth_segmentation_pred():
                        temp = tf.boolean_mask(self.predictions[othertask], self.mask[task])
                        return tf.expand_dims(temp[0, :, :, :], axis=0)
                    def getNoDepth_segmentation_pred():
                        return tf.zeros([1, self.opt.image_height, self.opt.image_width, self.opt.num_classes], dtype=tf.float32, name="No_Depth_segmentation_pred")
                    # Put a prediction OF THE OTHER TASK from the batch to the summary (if exists in the batch)
                    img2sum = tf.cond(self.has_smpl[task], getDepth_segmentation_pred, getNoDepth_segmentation_pred)
                    img2sum = tf.reshape(tf.cast(tf.argmax(img2sum, axis=-1), dtype=tf.float32),
                                         shape=[-1, self.opt.image_height, self.opt.image_width, 1])
                    tf.summary.image(task + '/pred_' + othertask, img2sum, max_outputs=1)
                    # Save the images to be written later
                    self.images2write[task + '_pred_' + othertask] = tf.squeeze(img2sum, axis=[0, 3])

        # Whatever the task is, put an input image from the batch to the summary
        def getOrig():
            return tf.expand_dims(tf.boolean_mask(self.batch_images, self.mask[task])[0, :, :, :], axis=0)
        def getNoOrig():
            return tf.zeros([1, self.opt.image_height, self.opt.image_width, 3], dtype=tf.float32, name="No_Orig")
        img2sum = tf.cond(self.has_smpl[task], getOrig, getNoOrig)
        tf.summary.image(task + '/input', img2sum, max_outputs=1)
        self.images2write[task + '_input'] = tf.squeeze(img2sum, axis=0)

    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    # Function dedicated to colelct summaries
    def collect_summaries(self):
        with tf.name_scope("Summaries"):
            # Display the total loss in the summary
            tf.summary.scalar('Total_Loss', self.total_loss)
            # Add the overall loss to the summarry
            tf.summary.scalar('Learning_rate', self.opt.learning_rate)
            # Add the task dependent stuff to the summary
            self.images2write = {}
            for task in self.Tasks:
                self.TaskDependent_Summary(task)

    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    # Function dedicated to save the network
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

    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    # Function dedicated to train MENet network
    def train(self):

        self.build_train_graph()
        self.collect_summaries()

        # Count the number of trainable scalars / variables in the model
        with tf.name_scope("ModelParamsFingerPrint"):
            self.modelNumDOF = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
            self.modelNumVars = len(tf.trainable_variables())

        # Define the saver
        self.saver = tf.train.Saver([var for var in tf.trainable_variables()] + \
                                    [self.global_step],
                                    save_relative_paths=True,
                                    max_to_keep=self.opt.max_model_saved)

        # Define the session superviser
        sv = tf.train.Supervisor(logdir=self.opt.logdir,
                                 save_summaries_secs=0,
                                 saver=None)


        # Actually runs the session
        with sv.managed_session(config=self.SessionConfig) as sess:

            # If found a remaining ckpt restore from this point
            if(os.path.isfile(self.opt.logdir + "/model.latest.meta")):
                print('Restoring from the latest Checkpoint')
                self.saver.restore(sess, self.opt.logdir + "model.latest")
                print('Done')

            # Activate debug when mode enabled
            if (self.opt.debug):
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            # Define the fetches
            fetches = {
                "train": self.train_op,
                "global_step": self.global_step,
                "incr_global_step": self.incr_global_step
            }

            print("(Scalar) trainable variables : (" + str(sess.run(self.modelNumDOF)) + ") " + str(
                self.modelNumVars))
            for step in range(int(self.opt.num_steps_per_epoch * self.opt.num_epochs)):
                start_time = time.time()

                # Define the Summary/Save/Display related fetches
                if step % self.opt.summary_freq == 0:
                    fetches["loss"] = self.total_loss
                    fetches["losses"] = self.losses
                    fetches["summary"] = sv.summary_op
                if self.opt.save_images and step % self.opt.save_model_freq == 0:
                    fetches["images2write"] = self.images2write

                # Run the network with the fetches
                results = sess.run(fetches)
                gs = results["global_step"]

                # Summary/Save/Display related stuff
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
                        # Check if the Images folder is setup
                        if not os.path.exists(self.opt.ImagesDirectory):
                            os.makedirs(self.opt.ImagesDirectory)
                        # Write images prediction vs GT
                        im2write = results["images2write"]
                        for name, img_tens in im2write.items():
                            if len(img_tens.shape)>2:
                                if img_tens.shape[2] == 3:
                                    img = Image.fromarray(np.uint8(255.0 * img_tens))
                                    img.save(os.path.join(self.opt.ImagesDirectory, str(gs) + "_" + name + ".jpeg"))
                                    continue

                            # The predicted images must be converted
                            if (img_tens.ptp()>0):
                                if 'depth' in name:
                                    img_tens = 255.0 * (img_tens - img_tens.min()) / (img_tens.ptp())
                                elif 'segmentation' in name:
                                    img_tens = 255.0 * img_tens / (self.opt.num_classes-1)
                            img_tens = np.uint8(img_tens)
                            img = Image.fromarray(img_tens)
                            img.save(os.path.join(self.opt.ImagesDirectory, str(gs) + "_" + name + ".jpeg"))
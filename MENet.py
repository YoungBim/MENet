from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import KPI
from tensorflow.python import debug as tf_debug
from PIL import Image
from random import shuffle
slim = tf.contrib.slim

from enet import ENetEncoder, ENetSegDecoder, ENetDepthDecoder, ENet_arg_scope
from get_class_weights import ENet_weighing, median_frequency_balancing


from losses import depth_loss_nL1, segmentation_loss_wce, depth_loss_nL1_Reg
from dataPreparation import datasetAsList, write_records_from_file, _parse_function

from functools import partial

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
        image_files, annotation_files = datasetAsList(dataset_dir=self.opt.dataset_dir, TaskDirs=self.TaskDirs, Tasks=self.Tasks)
        # If necessary write TF reccords
        if(self.opt.write_tfreccords):
            if not os.path.exists(self.opt.tf_rec_path):
                os.makedirs(self.opt.tf_rec_path)
            filelist = [f for f in os.listdir(self.opt.tf_rec_path) if f.endswith(".tfrecords")]
            for f in filelist:
                os.remove(os.path.join(self.opt.tf_rec_path, f))
            for task in self.Tasks:
                write_records_from_file(image_files[task], annotation_files[task], self.TaskLabel[task], task, self.opt.tf_rec_path, self.opt.num_tfreccords)
            exit()
        # Know the number steps to take before decaying the learning rate and batches per epoch
        self.opt.dataset_num_samples = {}
        self.opt.dataset_total_num_samples = 0
        for task in self.Tasks:
            self.opt.dataset_num_samples[task] = len(image_files[task])
            self.opt.dataset_total_num_samples = self.opt.dataset_total_num_samples + self.opt.dataset_num_samples[task]
        self.opt.num_samples_per_epoch = self.opt.dataset_total_num_samples
        self.opt.num_batches_per_epoch = self.opt.num_samples_per_epoch / self.opt.batch_size

        return image_files, annotation_files

    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    # Function dedicated to compute the class weighting
    def compute_class_weight(self, annotation_files):
        # =================CLASS WEIGHTS===============================
        # Median frequency balancing class_weights
        if self.opt.weighting == "MFB":
            self.class_weights = median_frequency_balancing(annotation_files["segmentation"])
        # Inverse weighing probability class weights
        elif self.opt.weighting == "ENET":
            self.class_weights = ENet_weigthing(annotation_files["segmentation"])

    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    # Function dedicated to queue data and batch samples
    def load_Data(self):
        filenames = [os.path.join(root, name).replace('\\','/')
                             for root, _, files in os.walk(self.opt.tf_rec_path, followlinks=True)
                             for name in files
                             if name.endswith(".tfrecords")]

        shuffle(filenames)
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.shuffle(buffer_size=500)
        dataset = dataset.repeat(self.opt.num_epochs)
        dataset = dataset.batch(self.opt.batch_size)
        dataset = dataset.map(partial(_parse_function, batch_size = self.opt.batch_size))
        dataset = dataset.prefetch(5)

        iterator = dataset.make_one_shot_iterator()
        mybatch = iterator.get_next()
        self.batch_images = mybatch['image']
        self.batch_annotations = mybatch['annotation']
        self.batch_tasks = mybatch['task']



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
        self.depth_smoothloss = {}
        if task == "segmentation":
            loss = segmentation_loss_wce(task, pred, anots, self.opt.num_classes, self.class_weights)
        elif task == "depth":
            self.depth_smoothloss['smooth_d1'], self.depth_smoothloss['smooth_d2'], self.depth_smoothloss['smooth_wsum'], self.depth_smoothloss['n1_loss'], loss = depth_loss_nL1_Reg(task, pred, anots)
        else:
            print("Please define a loss for this task")
            loss = tf.constant(-1, dtype=tf.float32)
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
            loss[task] = tf.multiply(loss[task],n_smpl[task])
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

    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    # Function dedicated to compute the loss from the inference predictions
    def Optimize(self):
        self.global_step = tf.train.get_or_create_global_step()
        # Compute the decay steps
        self.opt.decay_steps = int(self.opt.num_epochs_before_decay * self.opt.num_batches_per_epoch)

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
    def build_graph(self,Mode):

        # Get the data prepared
        image_files, annotation_files = self.prepare_Data()

        if Mode == 'Train':
            # Compute the wieghts to apply on the classes (segmentation task)
            self.compute_class_weight(annotation_files)
        elif Mode == 'Eval':
            self.class_weights = 0
            pass
        else:
            print('Mode not found')
            exit()

        with tf.name_scope("Data"):
            self.load_Data()

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

            def getDepth_smoothloss_d1():
                return self.depth_smoothloss['smooth_d1']
            def getNoDepth_smoothloss_d1():
                return tf.constant(0,dtype=tf.float32)
            temp = tf.cond(self.has_smpl[task], getDepth_smoothloss_d1, getNoDepth_smoothloss_d1)
            tf.summary.scalar(task + '/Loss_depth_smooth/d_1', temp)

            def getDepth_smoothloss_d2():
                return self.depth_smoothloss['smooth_d2']
            def getNoDepth_smoothloss_d2():
                return tf.constant(0,dtype=tf.float32)
            temp = tf.cond(self.has_smpl[task], getDepth_smoothloss_d2, getNoDepth_smoothloss_d2)
            tf.summary.scalar(task + '/Loss_depth_smooth/d_2', temp)

            def getDepth_smoothloss_wsum():
                return self.depth_smoothloss['smooth_wsum']
            def getNoDepth_smoothloss_wsum():
                return tf.constant(0,dtype=tf.float32)
            temp = tf.cond(self.has_smpl[task], getDepth_smoothloss_wsum, getNoDepth_smoothloss_wsum)
            tf.summary.scalar(task + '/Loss_depth_smooth/d_wsum', temp)

            def getDepth_n1loss():
                return self.depth_smoothloss['n1_loss']
            def getNoDepth_n1loss():
                return tf.constant(0,dtype=tf.float32)
            temp = tf.cond(self.has_smpl[task], getDepth_n1loss, getNoDepth_n1loss)
            tf.summary.scalar(task + '/Loss_depth_smooth/n1loss', temp)

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
        if step == 'latest':
            print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
            self.saver.save(sess,
                            os.path.join(checkpoint_dir, self.opt.model_name + '.latest'))
        else:
            self.saver.save(sess,
                            os.path.join(checkpoint_dir, self.opt.model_name),
                            global_step=step)

    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    # Function that prints usefull information for the user
    def printInitialInfo(self,sess):
        print("(Scalar) trainable variables : (" + str(sess.run(self.modelNumDOF)) + ") " + str(
            self.modelNumVars))
        pt = "Dataset of " + str(self.opt.dataset_total_num_samples) + " samples ("
        for task in self.Tasks:
            pt = pt + str(self.opt.dataset_num_samples[task]) + " for " + task + ", "
        pt = pt[:-2] + ")"
        print(pt)

    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    # Function dedicated to train MENet network
    def train(self):

        self.build_graph('Train')
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
            if(os.path.isfile(self.opt.logdir + self.opt.model_name + ".latest.meta")):
                print('Restoring from the latest Checkpoint')
                self.saver.restore(sess, self.opt.logdir + self.opt.model_name + ".latest")
                print('Done')

            # Activate debug when mode enabled
            if (self.opt.debug):
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            # Print uselfull info for user
            self.printInitialInfo(sess)

            # Define the fetches
            fetches = {
                "train": self.train_op,
                "global_step": self.global_step,
                "learning_rate": self.opt.learning_rate
            }

            for step in range(int(self.opt.num_batches_per_epoch * self.opt.num_epochs)):

                # Define the Summary/Save/Display related fetches
                if step % self.opt.summary_freq == 0:
                    fetches["loss"] = self.total_loss
                    fetches["losses"] = self.losses
                    fetches["summary"] = sv.summary_op
                if self.opt.save_images and (step + 1) % self.opt.save_model_freq == 0:
                    fetches["images2write"] = self.images2write


                # Run the network with the fetches
                start_time = time.time()
                results = sess.run(fetches)
                gs = results["global_step"]

                # Summary/Save/Display related stuff
                if step % self.opt.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil(gs / self.opt.num_batches_per_epoch)
                    train_step = gs - (train_epoch - 1) * self.opt.num_batches_per_epoch
                    print("Epoch: [%2d] [%5d/%5d] time: %4.2f/it learning rate: %1.6f" \
                            % (train_epoch, train_step, self.opt.num_batches_per_epoch, \
                                time.time() - start_time, results["learning_rate"]))
                    pt = "\t losses : "
                    for task in self.Tasks:
                        pt = pt + task + " : " + str(results["losses"][task]) + " | "
                    print(pt + "total : %.3f"%(results["loss"]))

                if (step + 1) % self.opt.save_model_freq == 0:
                    self.save(sess, self.opt.logdir, gs)
                    self.save(sess, self.opt.logdir, 'latest')
                    if self.opt.save_images:
                        # Check if the Images folder is setup
                        if not os.path.exists(self.opt.ImagesDirectory):
                            os.makedirs(self.opt.ImagesDirectory)
                        # Write images prediction vs GT
                        im2write = results["images2write"]
                        for name, img_tens in im2write.items():
                            if len(img_tens.shape)>2:
                                if img_tens.shape[2] == 3:
                                    img = Image.fromarray(np.uint8(img_tens))
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

    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    # Function dedicated to evaluate MENet network
    def evaluate(self):

        self.build_graph('Eval')

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

        # Initialize the KPI object
        My_KPI = KPI.KPI(self.Tasks, self.opt.num_classes, 100)

        # Actually runs the session
        with sv.managed_session(config=self.SessionConfig) as sess:

            # If found a remaining ckpt restore from this point
            if(os.path.isfile(self.opt.logdir + self.opt.model_name + ".latest.meta")):
                print('Restoring from the latest Checkpoint')
                self.saver.restore(sess, self.opt.logdir + self.opt.model_name + ".latest")
                print('Done')
            else:
                print('Couldn''t restore the checkpoint')
                exit()

            # Define the fetches
            fetches = {
                "predictions": self.pred,
                "anotations": self.anots
            }

            # Get the samples processed
            for step in range(int(self.opt.num_batches_per_epoch * self.opt.num_epochs)):

                start_time = time.time()
                results = sess.run(fetches)
                time_per_frame_ms = int((time.time() - start_time)*1000/self.opt.batch_size)
                print('Forward per frame ' + str(time_per_frame_ms) + ' ms')
                for key in results['predictions'].keys():
                    anot = results['anotations'][key]
                    pred = results['predictions'][key]

                    if key == 'segmentation':
                        My_KPI.update_segKPI(anot,pred)
                    elif key == 'depth':
                        My_KPI.update_depthKPI(anot,pred)
                    else:
                        print('KPI not available for this task')
                print('Eval step ' +  str(step) + ' on ' + str(int(self.opt.num_batches_per_epoch * self.opt.num_epochs)))
            My_KPI.compute_KPIs(os.path.join(self.opt.logdir,'KPIs.pkl'))
            KPIs = My_KPI.parse_KPIs(os.path.join(self.opt.logdir,'KPIs.pkl'))
            print(KPIs)




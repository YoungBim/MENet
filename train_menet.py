import os
import tensorflow as tf
import pprint
import numpy as np
import random
from MENet import MENet

## https://stackoverflow.com/questions/37893755/tensorflow-set-cuda-visible-devices-within-jupyter
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
## https://github.com/tensorflow/tensorflow/issues/7778
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

Mode = 'Xdebug'

#==============INPUT ARGUMENTS==================
logdirectory = "./log/"
datasetdirectory = "./dataset/"

flags = tf.app.flags
#Directory arguments
flags.DEFINE_string('dataset_dir', datasetdirectory , 'The dataset directory to find the train, validation and test images.')
flags.DEFINE_string('logdir', logdirectory + 'debug', 'The log directory to save your checkpoint and event files.')
flags.DEFINE_integer("summary_freq", 10, "Logging every log_freq iterations")
flags.DEFINE_integer("save_model_freq", 100, "Logging every log_freq iterations")
flags.DEFINE_integer("max_model_saved", 5, "Maximum number of model saved")


#Training arguments
flags.DEFINE_integer('num_classes', 12, 'The number of classes to predict.')
flags.DEFINE_integer('train_val_rate', 0.05, 'The number of classes to predict.')
flags.DEFINE_integer('batch_size', 10, 'The batch_size for training.')
flags.DEFINE_integer('eval_batch_size', 10, 'The batch size used for validation.')
flags.DEFINE_integer('image_height', 360, "The input height of the images.")
flags.DEFINE_integer('image_width', 480, "The input width of the images.")
flags.DEFINE_integer('num_epochs', 300, "The number of epochs to train your model.")
flags.DEFINE_integer('num_epochs_before_decay', 100, 'The number of epochs before decaying your learning rate.')
flags.DEFINE_float('weight_decay', 2e-4, "The weight decay for ENet convolution layers.")
flags.DEFINE_float('learning_rate_decay_factor', 1e-1, 'The learning rate decay factor.')
flags.DEFINE_float("adam_momentum", 0.9, "Momentum term of adam (beta1)")
flags.DEFINE_float('initial_learning_rate', 5e-4, 'The initial learning rate for your training.')
flags.DEFINE_string('weighting', "MFB", 'Choice of Median Frequency Balancing or the custom ENet class weights.')

Tasks = ["segmentation", "depth"]
TaskDirs = {Tasks[0]: 'seg', Tasks[1]: 'depth'}
TaskLabel = {Tasks[i]: np.uint8(i) for i in range(len(Tasks))}

#Architectural changes
flags.DEFINE_integer('num_initial_blocks', 1, 'The number of initial blocks to use in ENet.')
flags.DEFINE_integer('stage_two_repeat', 2, 'The number of times to repeat stage two.')
flags.DEFINE_boolean('skip_connections', True, 'If True, perform skip connections from encoder to decoder.')

FLAGS = flags.FLAGS

def main(_):
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)

    My_MENet = MENet(FLAGS, Tasks, TaskDirs, TaskLabel)
    My_MENet.train()


if __name__ == '__main__':
    tf.app.run()

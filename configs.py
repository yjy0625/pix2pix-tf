import tensorflow as tf

# limit to only one gpu
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1" for multiple

flags = tf.app.flags
# training params
flags.DEFINE_integer("epoch", 250, "Number of epochs to train. [25]")
flags.DEFINE_float("lr_d", 1e-8, "Learning rate for discriminator [1e-8]")
flags.DEFINE_float("lr_g", 1e-4, "Learning rate for generator [1e-4]")
flags.DEFINE_float("lam", 100, "Lambda value for g_l1_loss [100]")
flags.DEFINE_integer("batch_size", 64, "Number of images in batch [64]")
# dataset params
flags.DEFINE_string("data_dir", "data", "Path to datasets directory [data]")
flags.DEFINE_string("dataset", "facades", "The name of dataset [facades]")
# flags for running
flags.DEFINE_string("version", "v2", "Name of experiment for current run [experiment]")
flags.DEFINE_boolean("train", True, "Train if True, otherwise test [False]")
flags.DEFINE_boolean("use_gpu", True, "Use GPU for if True, otherwise use CPU [True]")
# directory params
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Path to save the checkpoint data [checkpoint]")
flags.DEFINE_string("log_dir", "logs", "Path to log for TensorBoard [logs]")
flags.DEFINE_string("output_dir", "output", "Path to store output images [output]")
flags.DEFINE_string("image_ext", "jpg", "Image extension to find [jpg]")
FLAGS = flags.FLAGS

import tensorflow as tf

flags = tf.app.flags
# training params
flags.DEFINE_integer("epoch", 250, "Number of epochs to train. [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate for Adam optimizer [0.0002]")
flags.DEFINE_integer("batch_size", 64, "Number of images in batch [64]")
# dataset params
flags.DEFINE_string("data_dir", "data", "Path to datasets directory [data]")
flags.DEFINE_string("dataset", "facades", "The name of dataset [facades]")
# flags for running
flags.DEFINE_string("version", "v1", "Name of experiment for current run [experiment]")
flags.DEFINE_boolean("train", True, "Train if True, otherwise test [False]")
flags.DEFINE_boolean("use_gpu", True, "Use GPU for if True, otherwise use CPU [True]")
# directory params
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Path to save the checkpoint data [checkpoint]")
flags.DEFINE_string("log_dir", "logs", "Path to log for TensorBoard [logs]")
flags.DEFINE_string("output_dir", "output", "Path to store output images [output]")
flags.DEFINE_string("image_ext", "jpg", "Image extension to find [jpg]")
FLAGS = flags.FLAGS

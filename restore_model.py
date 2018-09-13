import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

def get_restorer(FLAGS):
    checkpoint_dir_path = os.path.join('./', FLAGS.checkpoint_dir, FLAGS.version)
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir_path)
    restorer = None
    if checkpoint_path != None:
        if FLAGS.restore:
            print('----- Restoring from Model -----')
            model_variables = slim.get_model_variables()
            restorer = tf.train.Saver(model_variables)
        else:
            restorer = tf.train.Saver()
        print("Model restored from :", checkpoint_path)
    else:
        if not os.path.exists(checkpoint_dir_path):
            os.makedirs(checkpoint_dir_path)
        print("Model not found. Training from scratch.")

    return restorer, checkpoint_path

import os
import tensorflow as tf
import numpy as np
import sys
import time
import PIL.Image as Image
from configs import FLAGS
from model import Pix2pix
from data_util import BatchLoader
import restore_model
import logz

def mkdir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

def main():
	p2p = Pix2pix(FLAGS)

	if FLAGS.train:

		img_loader = BatchLoader(FLAGS, 'train')
		max_steps = FLAGS.epoch * img_loader.get_num_steps_per_epoch()

		# summary
		tf.summary.image('img/input', p2p.g_input)
		tf.summary.image('img/real_output', p2p.real_output)
		tf.summary.image('img/fake_output', p2p.fake_output)
		tf.summary.scalar('losses/g_loss', p2p.loss_g)
		tf.summary.scalar('losses/d_loss', p2p.loss_d)
		tf.summary.scalar('losses/total_loss', p2p.total_loss)

		# summary and init operations
		summary_op = tf.summary.merge_all()
		init_op = tf.group(
			tf.global_variables_initializer(),
			tf.local_variables_initializer()
		)

		# get model restorer and saver
		restorer, restore_ckpt = restore_model.get_restorer(FLAGS)
		saver = tf.train.Saver(max_to_keep=10)

		# initialize Tensorflow session and start training
		config = tf.ConfigProto(allow_soft_placement = True)
		with tf.Session(config = config) as sess:
			# initialize all variables
			print("Initializing...")
			sess.run(init_op)

			# restore most recent checkpoint, train from scratch if none exists
			if not restorer is None:
				restorer.restore(sess, restore_ckpt)
				print("Model successfully restored.")
			else:
				print("Training from scratch.")

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess, coord)

			# configure summary writer
			summary_path = os.path.join('./', FLAGS.log_dir, FLAGS.version)
			mkdir(summary_path)
			summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)

			# get current step of the model
			curr_step = sess.run(p2p.global_step)

			# main training loop
			print("Main training loop starts.")
			for step in range(max_steps):
				# load next batch to prepare for training
				img_in, img_real_out = img_loader.load_batch()

				# prepare input matrices to be fed into the network
				feed_dict = {
					p2p.g_input: img_in,
					p2p.real_output: img_real_out
				}

				# start timing runtime
				start = time.time()

				# update D network
				_, loss_d = sess.run([p2p.optimizer_d, p2p.loss_d], feed_dict=feed_dict)

				# update G network twice for stability
				sess.run(p2p.optimizer_g, feed_dict=feed_dict)
				_, loss_g = sess.run([p2p.optimizer_g, p2p.loss_g], feed_dict=feed_dict)

				# get loss
				loss = loss_d + loss_g

				# stop timing
				end = time.time()

				# print training progress every 10 steps
				if step % 10 == 0:
					logz.log_tabular("Step", step)
					logz.log_tabular("D Loss", loss_d)
					logz.log_tabular("G Loss", loss_g)
					logz.log_tabular("Total Loss", loss)
					logz.log_tabular("Time Spent", end - start)
					logz.dump_tabular()

				# add summary for tensorboard every 50 steps
				if step % 50 == 0:
					summary_str = sess.run(summary_op, feed_dict=feed_dict)
					summary_writer.add_summary(summary_str, step)
					summary_writer.flush()

				# store checkpoint in file once in a while
				if (step > 0 and step % 1000 == 0) or (step == max_steps - 1):
					save_dir = os.path.join('./', FLAGS.checkpoint_dir, FLAGS.version)
					mkdir(save_dir)

					save_ckpt = os.path.join('./', save_dir, 'p2p_'+str(_global_step)+'.ckpt')
					saver.save(sess, save_ckpt)
					print('Weights have been saved')
					
	else:
		img_loader = BatchLoader(FLAGS, 'test')

		with tf.Session() as sess:
			init_op = tf.group(
				tf.global_variables_initializer(),
				tf.local_variables_initializer()
			)
			sess.run(init_op)

			if not restorer is None:
				restorer.restore(sess, restore_ckpt)
				print("Model successfully restored.")
			else:
				sys.exit('Trained model not found.')

				img_in, _ = img_loader.load_batch()
				output = sess.run(p2p.fake_output, feed_dict={p2p.g_input: img_in})

				save_count = 0
				for img in output:
					save_count += 1
					save_path = os.path.join('./', FLAGS.output_dir, str(save_count) + '.' + FLAGS.image_ext) 
					cv2.imwrite(save_path, img)
				
				print("Done!")

if __name__ == '__main__':
	main()

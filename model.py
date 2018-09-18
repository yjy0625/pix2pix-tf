import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class Pix2pix(object):

	def __init__(self, FLAGS):
		'''
			Builds the GAN model, including generator, discriminator, losses, and optimizers.
		'''
		self.global_step = tf.Variable(0, name='global_step', trainable=False)
		self.use_gpu = FLAGS.use_gpu

		self.g_input = tf.placeholder(tf.float32, [None, 256, 256, 3], 'g_input')

		self.real_output = tf.placeholder(tf.float32, [None, 256, 256, 3], 'real_output')
		self.fake_output = self.generator(self.g_input)

		self.d_real = self.discriminator(self.g_input, self.real_output)
		self.d_fake = self.discriminator(self.g_input, self.fake_output, reuse=True)

		eps = 1e-10
		self.loss_d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real, labels=tf.ones_like(self.d_real)))
		self.loss_d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake, labels=tf.zeros_like(self.d_fake)))
		self.loss_d = tf.multiply(0.5, (self.loss_d_real + self.loss_d_fake), name='loss_d')
		self.loss_g_gan = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake, labels=tf.ones_like(self.d_fake)))
		self.loss_g_l1 = tf.reduce_mean(tf.abs(self.real_output - self.fake_output))
		self.loss_g = tf.add(self.loss_g_gan, FLAGS.lam * self.loss_g_l1, name='loss_g')

		self.total_loss = tf.add(self.loss_d, self.loss_g, name='total_loss')
		
		self.vars_d = [var for var in tf.trainable_variables() if var.name.startswith("D")]
		self.vars_g = [var for var in tf.trainable_variables() if var.name.startswith("G")]

		self.optimizer_d = tf.train.AdamOptimizer(FLAGS.lr_d, beta1=0.5) \
			.minimize(self.loss_d, var_list=self.vars_d)
		self.optimizer_g = tf.train.AdamOptimizer(FLAGS.lr_g, beta1=0.5) \
			.minimize(self.loss_d, var_list=self.vars_g, global_step=self.global_step)

	def generator(self, input):
		'''
			Generator that takes in image input and generates transferred image
			as output.

			arguments:
				input: input image of size (256, 256, 3)

			returns:
				output: output image of the same size
		'''
		filter_counts = [3, 64, 128, 256, 512, 512, 512, 512, 512]

		with tf.variable_scope("G"):
			encoders = []
			encoders.append(input)
			with slim.arg_scope([slim.conv2d], 
								stride=2,
								padding='SAME',
								activation_fn=tf.nn.relu,
								normalizer_fn=slim.batch_norm):
				for i in range(8):
					z = slim.conv2d(encoders[-1], filter_counts[i + 1], [4, 4], scope='g-conv{}'.format(i + 1))
					encoders.append(z)

			decoders = []
			decoders.append(encoders[-1])
			with slim.arg_scope([slim.conv2d_transpose],
								stride=2,
								padding='SAME',
								normalizer_fn=slim.batch_norm):
				for i in range(8):
					if i < 7:
						activation_fn = tf.nn.relu
					else:
						activation_fn = tf.nn.tanh

					z = slim.conv2d_transpose(decoders[-1], filter_counts[7 - i], [4, 4], activation_fn=activation_fn, scope='g-deconv{}'.format(8 - i))
					if i < 3:
						z = slim.dropout(z, 0.5, scope='g-deconv{}-dropout'.format(8-i))
					if i < 7:
						z = tf.concat((z, encoders[7 - i]), axis=3, name='g-deconv{}-concat'.format(8 - i))
					decoders.append(z)

		return decoders[-1]

	def discriminator(self, input, output, reuse=False):
		'''
			Descriminator.

			arguments:
				input: input image for the unknown program
				output: output generated by the unknown program

			output:
				guess: a symbolic scalar that predicts whether output is real (1) or fake (0)
		'''
		with tf.variable_scope("D"):
			z = tf.concat((input, output), axis=3, name='d-concat')
			with slim.arg_scope([slim.conv2d],
								reuse=reuse,
								padding='SAME',
								activation_fn=tf.nn.relu,
								normalizer_fn=slim.batch_norm):
				with slim.arg_scope([slim.conv2d], stride=2):
					z = slim.stack(z, slim.conv2d, [(6, [4, 4]), (64, [4, 4]), (128, [4, 4]), (256, [4, 4])], scope='d-conv1')

				with slim.arg_scope([slim.conv2d], stride=1):
					z = slim.stack(z, slim.conv2d, [(512, [4, 4]), (1, [4, 4])], scope='d-conv2')

				z = slim.flatten(z, scope='d-flatten')
				z = slim.fully_connected(z, 1, reuse=reuse, activation_fn=None, scope='d-fc')

		return z

	def test_model(self):
		init = tf.initialize_all_variables()

		with tf.Session() as sess:
			sess.run(init)
			summary_writer = tf.summary.FileWriter(os.getcwd(), graph=sess.graph)

if __name__ == "__main__":
	flags = tf.app.flags
	flags.DEFINE_float("lr_d", 0.0002, "")
	flags.DEFINE_float("lr_g", 0.0002, "")

	device = '/cpu:0'
	with tf.device(device):
		dcgan = Pix2pix(flags.FLAGS)
		dcgan.test_model()


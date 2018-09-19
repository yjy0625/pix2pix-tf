import os
import numpy as np
import cv2

class BatchLoader(object):
	def __init__(self, FLAGS, usage):
		self.image_dir = FLAGS.data_dir
		self.dataset = FLAGS.dataset
		self.image_ext = FLAGS.image_ext
		self.batch_size = FLAGS.batch_size
		self.usage = usage

		self.curr_batch = 0

		self.prepare_images()

	def read_image_left_half(self, filename):
		img = cv2.imread(filename)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = img.astype(np.float32) / 127.5 - 1.0
		return img[:, :int(img.shape[1]/2 + 0.5), :]

	def read_image_right_half(self, filename):
		img = cv2.imread(filename)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = img.astype(np.float32)/ 127.5 - 1.0
		return img[:, int(img.shape[1]/2 + 0.5):, :]

	def save_image(self, img):
		cv2.imwrite("save.jpg", img)

	def prepare_images(self):
		image_path = os.path.join(os.getcwd(), self.image_dir, self.dataset, self.usage)
		filenames = os.listdir(image_path)
		filenames = list(filter(lambda x: x.endswith('.' + self.image_ext), filenames))
		
		img_inputs = np.array([np.array([self.read_image_right_half(os.path.join(image_path, fname)) for fname in filenames])])[0]
		img_outputs = np.array([np.array([self.read_image_left_half(os.path.join(image_path, fname)) for fname in filenames])])[0]

		shuffle_index = np.random.permutation(img_inputs.shape[0])
		self.img_inputs = img_inputs[shuffle_index]
		self.img_outputs = img_outputs[shuffle_index]

	def load_images(self):
		return self.img_inputs, self.img_outputs

	def get_num_steps_per_epoch(self):
		return int(self.img_inputs.shape[0] / self.batch_size + 0.5)

	def load_batch(self):
		start_ix = (self.curr_batch * self.batch_size) % self.img_inputs.shape[0]
		end_ix = start_ix + self.batch_size
		self.curr_batch += 1

		return self.img_inputs[0:self.batch_size], self.img_outputs[0:self.batch_size]
		# return self.img_inputs[start_ix:end_ix], self.img_outputs[start_ix: end_ix]

	def test_class(self):
		iin, iout = self.load_images()
		print("num of steps per epoch: {}".format(self.get_num_steps_per_epoch()))
		for i in range(3000):
			_, _ = self.load_batch()
			if i % 1000 == 0:
				print('curr_batch: {}'.format(self.curr_batch))
		print("image shape: {}".format(self.img_inputs[0].shape))
		self.save_image(self.img_inputs[0])

if __name__ == '__main__':
	import configs
	bl = BatchLoader(configs.FLAGS, 'train')
	bl.test_class()

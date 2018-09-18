import numpy as np
import torch 

from net import Net
from generator import Generator

import utils
from PIL import Image

SIZE = (30, 30)
MARGIN = 1
NOISE = 0.1
LENGTH = 3

# DEVICE = torch.device('cuda:0')
DEVICE = torch.device('cpu')

np.set_printoptions(threshold=np.inf, precision=4, suppress=True, linewidth=160)

gen = Generator(LENGTH, SIZE, MARGIN, NOISE)
net = Net(DEVICE)

net.load_state_dict( torch.load('model', map_location={'cuda:0': 'cpu'}) )
print(net)

#---- eval
def get_batch(size):
	batch_x = []
	batch_y = []
	batch_img = []

	for i in range(size):
		chars, img = gen.generate()
		chars = list( map(lambda x: ord(x) - ord('A'), chars) )

		batch_x.append(np.array(img, dtype=np.float32))
		batch_y.append(chars)
		batch_img.append(img)

	batch_x = np.array(batch_x) / 255.
	batch_y = np.array(batch_y)

	return batch_x, batch_y, batch_img

def to_txt(chars):
	chars = list( map(lambda x: chr(x + ord('A')), chars) )
	return "".join(chars)

test_x, test_y, test_img = get_batch(1024)

pred_y, pred_mask = net(test_x, LENGTH)
pred_y = pred_y.argmax(dim=2).detach().cpu().numpy()
pred_mask = pred_mask.detach().cpu().numpy()

for i in range(1024):
	print("TASK: {} PRED: {}".format( to_txt(test_y[i]), to_txt(pred_y[i] )))

	for l in range(3):
		img = test_img[i]
		msk = pred_mask[i, l]

		img = Image.fromarray(img)
		img = img.convert('RGB')
		img = np.array(img)

		img[:, :, 2] = (msk * 255).astype(np.uint8)

		# print(img)
		# exit()

		img = Image.fromarray(img)
		img = img.resize( (120, 120) )
		img.show()

	input()	
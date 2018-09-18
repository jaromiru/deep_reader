import numpy as np
import torch 

from net import Net
from generator import Generator

import utils
from PIL import Image

SIZE = (40, 40)
MARGIN = 1
NOISE = 0.1
MAX_LENGTH = 5

# DEVICE = torch.device('cuda:0')
DEVICE = torch.device('cpu')

np.set_printoptions(threshold=np.inf, precision=4, suppress=True, linewidth=160)

gen = Generator(MAX_LENGTH, SIZE, MARGIN, NOISE)
net = Net(DEVICE) 

net.load_state_dict( torch.load('model', map_location={'cuda:0': 'cpu'}) )
print(net)

#---- eval
def get_batch(size):
	batch_x = []
	batch_y = []
	batch_img = []

	for i in range(size):
		chars, img, lm = gen.generate()
		chars = list( map(lambda x: ord(x) - ord('A'), chars) )

		batch_x.append(img)
		batch_y.append(chars)
		batch_img.append(img)

	batch_x = np.array(batch_x, dtype=np.float32) / 255.
	batch_y = np.array(batch_y)

	return batch_x, batch_y, batch_img

def to_txt(chars):
	chars = list( map(lambda x: chr(x + ord('A')), chars) )
	return "".join(chars)

test_x, test_y, test_img = get_batch(8)

pred_y, pred_mask = net(test_x, MAX_LENGTH)
pred_y = pred_y.argmax(dim=2).detach().cpu().numpy()
pred_mask = pred_mask.detach().cpu().numpy()
pred_mask = np.insert(pred_mask, 0, np.ones(SIZE), axis=1)

for i in range(8):
	print("TASK: {} PRED: {}".format( to_txt(test_y[i]), to_txt(pred_y[i] )))

	imgs = []

	for l in range(MAX_LENGTH+1):
		img = test_img[i]
		msk = pred_mask[i, l]

		img = Image.fromarray(img)
		img = img.convert('RGB')
		img = np.array(img)

		img[:, :, 2] = (msk * 255).astype(np.uint8)

		imgs.append(img)

	imgs = np.concatenate(imgs, axis=1)
	imgs = Image.fromarray(imgs)
	imgs = imgs.resize( (120*(MAX_LENGTH+1), 120) )

	imgs.save('img/vis_' + str(i) + '.gif')
	# imgs.show()
	# input()	
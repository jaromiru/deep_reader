import numpy as np
import torch 

from net import Net
from generator import Generator

import utils

EPOCHS = 100000
BATCH_SIZE = 32
LR = 1e-3
LR_STEP = 0.1
LR_FAILS = 3

SIZE = (30, 30)
MARGIN = 1
NOISE = 0.
LENGTH = 3

DEVICE = torch.device('cuda:0')
# DEVICE = torch.device('cpu')

np.set_printoptions(threshold=np.inf, precision=4, suppress=True, linewidth=160)

gen = Generator(LENGTH, SIZE, MARGIN, NOISE)
net = Net(DEVICE)

print(net)

#---- train
def get_batch(size):
	batch_x = []
	batch_y = []
	for i in range(size):
		chars, img = gen.generate()
		chars = list( map(lambda x: ord(x) - ord('A'), chars) )

		batch_x.append(img)
		batch_y.append(chars)

	batch_x = np.array(batch_x, dtype=np.float32) / 255.
	batch_y = np.array(batch_y)

	return batch_x, batch_y

test_x, test_y = get_batch(1024)
test_loss = net.get_loss(test_x, test_y, LENGTH)

lr = LR
losses = []
best_loss = 1e6
lr_fails = 0

net.set_lr(lr)
print("LR: {:.2e}".format(lr))

for e in range(EPOCHS):
	train_x, train_y = get_batch(BATCH_SIZE)
	net.train(train_x, train_y, LENGTH)

	if utils.is_time(e, 100):
		pred_y, msks = net(test_x, LENGTH)
		pred_y = pred_y.argmax(dim=2).detach().cpu().numpy()

		corr = np.all(pred_y == test_y, 1).mean()
		test_loss = net.get_loss(test_x, test_y, LENGTH).item()	

		print("Epoch {}: loss {:.4f}, corr: {:.2f}".format(e, test_loss, corr))
		losses.append(test_loss)

		if test_loss > best_loss:
			lr_fails += 1
			print("." * lr_fails)
			
			if lr_fails >= LR_FAILS:
				lr = lr * LR_STEP
				net.set_lr(lr)
				print("LR: {:.2e}".format(lr))
		else:
			best_loss = test_loss
			lr_fails = 0

	if utils.is_time(e, 1000):
		torch.save(net.state_dict(), 'model')


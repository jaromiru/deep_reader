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

SIZE = (40, 40)
MARGIN = 1
NOISE = 0.1
MAX_LENGTH = 5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.set_printoptions(threshold=np.inf, precision=4, suppress=True, linewidth=160)

gen = Generator(MAX_LENGTH, SIZE, MARGIN, NOISE)
net = Net(DEVICE)

print(net)

#---- train
def get_batch(size):
    batch_x = []
    batch_y = []

    batch_lm = np.ones((size, MAX_LENGTH), dtype=np.float32)    # loss mask

    for i in range(size):
        chars, img, ln = gen.generate()
        chars = list(map(lambda x: ord(x), chars))
        chars = np.array(chars)

        batch_x.append(img)
        batch_y.append(chars)
        batch_lm[i, ln+1:] = 0

    batch_x = np.array(batch_x, dtype=np.float32) / 255
    batch_y = np.array(batch_y, dtype=np.int64) - ord('A')

    return batch_x, batch_y, batch_lm

test_x, test_y, test_lm = get_batch(1024)

lr = LR
losses = []
best_loss = 1e6
lr_fails = 0

net.set_lr(lr)
print("LR: {:.2e}".format(lr))

fps = utils.Fps()
fps.start()

for e in range(EPOCHS):
    train_x, train_y, train_lm = get_batch(BATCH_SIZE)
    net.train(train_x, train_y, MAX_LENGTH, train_lm)

    if utils.is_time(e, 100):
        pred_y, msks = net(test_x, MAX_LENGTH)
        pred_y = pred_y.argmax(dim=2).detach().cpu().numpy()

        cond = np.logical_or( (pred_y == test_y), (1 - test_lm) )
        corr = np.all(cond, 1).mean()
        test_loss = net.get_loss(test_x, test_y, MAX_LENGTH, test_lm).item() 

        print("Epoch {}: loss {:.3f}, corr: {:.0f}%, fps: {:.1f}".format(e, test_loss, corr * 100, fps.fps(e)))
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


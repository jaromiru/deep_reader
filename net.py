import torch
import torch.nn.functional as F

HIDDEN_DIM = 256
ACTION_DIM = 26 + 1 # one ending char '['

class Net(torch.nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()

        self.device = device

        self.c1 = torch.nn.Conv2d(1,  16, 3)
        self.c2 = torch.nn.Conv2d(16, 16, 3)
        self.c3 = torch.nn.Conv2d(16, 16, 3)

        self.rnn1 = torch.nn.LSTM(18496, HIDDEN_DIM)

        self.msk = torch.nn.Linear(HIDDEN_DIM, 1600)
        self.act = torch.nn.Linear(HIDDEN_DIM, ACTION_DIM)    

        self.loss_cross = torch.nn.CrossEntropyLoss(reduction='none')
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)

        self.to(self.device)
        
    def forward(self, batch, length):      
        batch_len = batch.shape[0]
        # acts = torch.zeros(batch_len, length, dtype=torch.int32, device=self.device)
        acts = torch.zeros(batch_len, length, ACTION_DIM, dtype=torch.float32, device=self.device)
        msks = torch.zeros(batch_len, length, 40, 40, dtype=torch.float32, device=self.device)

        batch = torch.from_numpy(batch).to(self.device)        
        msk = torch.ones(batch_len, 40, 40, device=self.device)
        hidden = None

        for i in range(length):
            x = batch * msk

            x = x.reshape(batch_len, 1, 40, 40) # 1 layer          

            x = F.relu( self.c1(x) )
            x = F.relu( self.c2(x) )
            x = F.relu( self.c3(x) )

            x = x.view(1, batch_len, 18496)

            x, hidden = self.rnn1(x, hidden)

            msk = F.sigmoid( self.msk(x) ).view(batch_len, 40, 40)
            act = self.act(x).view(batch_len, ACTION_DIM)

            # acts[:, i] = torch.argmax(act, dim=1)
            acts[:, i] = act
            msks[:, i] = msk

        return acts, msks

    def get_loss(self, train_x, train_y, length, loss_mask):
        batch_len = train_y.shape[0]

        acts, msks = self(train_x, length)
        acts = acts.view(batch_len * length, ACTION_DIM)

        train_y = torch.from_numpy(train_y.flatten()).to(self.device)
        loss_mask = torch.from_numpy(loss_mask.flatten()).to(self.device)

        loss = self.loss_cross(acts, train_y) * loss_mask
        return loss.mean()

    def train(self, train_x, train_y, length, loss_mask):
        loss = self.get_loss(train_x, train_y, length, loss_mask)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def set_lr(self, lr):
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr

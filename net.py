import torch
import torch.nn.functional as F

HIDDEN_DIM = 128
ACTION_DIM = 26

class Net(torch.nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()

        self.device = device

        self.c1 = torch.nn.Conv2d(1, 8, 3)
        self.c2 = torch.nn.Conv2d(8, 8, 3)
        self.c3 = torch.nn.Conv2d(8, 8, 3)

        self.rnn1 = torch.nn.LSTM(4608, HIDDEN_DIM)

        self.msk = torch.nn.Linear(HIDDEN_DIM, 900)
        self.act = torch.nn.Linear(HIDDEN_DIM, ACTION_DIM)    

        self.loss_cross = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)

        self.to(self.device)
        
    def forward(self, batch, length):      
        batch_len = batch.shape[0]
        # acts = torch.zeros(batch_len, length, dtype=torch.int32, device=self.device)
        acts = torch.zeros(batch_len, length, ACTION_DIM, dtype=torch.float32, device=self.device)
        msks = torch.zeros(batch_len, length, 30, 30, dtype=torch.float32, device=self.device)

        batch = torch.from_numpy(batch).to(self.device)        
        msk = torch.ones(batch_len, 30, 30, device=self.device)
        hidden = None

        for i in range(length):
            x = batch * msk

            x = x.reshape(batch_len, 1, 30, 30) # 1 layer          

            x = F.relu( self.c1(x) )
            x = F.relu( self.c2(x) )
            x = F.relu( self.c3(x) )

            x = x.view(1, batch_len, 4608)

            x, hidden = self.rnn1(x, hidden)

            msk = F.sigmoid( self.msk(x) ).view(batch_len, 30, 30)
            act = self.act(x).view(batch_len, ACTION_DIM)

            # acts[:, i] = torch.argmax(act, dim=1)
            acts[:, i] = act
            msks[:, i] = msk

        return acts, msks

    def get_loss(self, train_x, train_y, length):
        batch_len = train_y.shape[0]

        acts, msks = self(train_x, length)
        acts = acts.view(batch_len * length, ACTION_DIM)

        train_y = torch.from_numpy(train_y.flatten()).to(self.device)

        # from torchviz import make_dot

        # dot = make_dot(acts.mean(), params=dict(self.named_parameters()))
        # dot.format = 'svg'
        # dot.render()
        # exit()

        # print(acts)
        # print(train_y)

        loss = self.loss_cross(acts, train_y)
        return loss

    def train(self, train_x, train_y, length):
        loss = self.get_loss(train_x, train_y, length)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def set_lr(self, lr):
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr

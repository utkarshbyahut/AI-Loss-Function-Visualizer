import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=8):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        return nn.Sigmoid()(self.fc2(x))

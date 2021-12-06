import torch
import torch.nn as nn
import torch.nn.functional as F
import slimevolleygym

class MyModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, action_size)

    def forward(self, x):
        # TODO YOUR CODE HERE FOR THE FORWARD PASS
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
#         x = F.one_hot(x)
        return x

    def select_action(self, state):
        self.eval()
        x = self.forward(state)
        self.train()
        return x.max(1)[1].view(1, 1).to(torch.long)

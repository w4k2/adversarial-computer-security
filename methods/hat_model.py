import torch
import numpy as np


def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))


class HATModel(torch.nn.Module):

    def __init__(self, classes_per_task, size, wide=1):
        super().__init__()

        ncha = 3
        self.wide = wide

        self.c1 = torch.nn.Conv2d(ncha, int(64 * wide), kernel_size=size//8)
        s = compute_conv_output_size(size, size//8)
        s = s//2
        self.c2 = torch.nn.Conv2d(int(64*wide), int(128*wide), kernel_size=size // 10)
        s = compute_conv_output_size(s, size//10)
        s = s//2
        self.c3 = torch.nn.Conv2d(int(128*wide), int(256*wide), kernel_size=2)
        s = compute_conv_output_size(s, 2)
        s = s//2
        self.smid = s
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()

        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(int(256 * wide) * self.smid * self.smid, int(2048*wide))
        self.fc2 = torch.nn.Linear(int(2048 * wide), int(2048 * wide))
        self.last = torch.nn.ModuleList()
        for num_classes in classes_per_task:
            self.last.append(torch.nn.Linear(int(2048 * wide), num_classes))

        self.gate = torch.nn.Sigmoid()
        # All embedding stuff should start with 'e'
        num_tasks = len(classes_per_task)
        self.ec1 = torch.nn.Embedding(num_tasks, int(64 * wide))
        self.ec2 = torch.nn.Embedding(num_tasks, int(128 * wide))
        self.ec3 = torch.nn.Embedding(num_tasks, int(256 * wide))
        self.efc1 = torch.nn.Embedding(num_tasks, int(2048 * wide))
        self.efc2 = torch.nn.Embedding(num_tasks, int(2048 * wide))
        """ (e.g., used in the compression experiments)
        lo,hi=0,2
        self.ec1.weight.data.uniform_(lo,hi)
        self.ec2.weight.data.uniform_(lo,hi)
        self.ec3.weight.data.uniform_(lo,hi)
        self.efc1.weight.data.uniform_(lo,hi)
        self.efc2.weight.data.uniform_(lo,hi)
        # """

        return

    def forward(self, x, t, s=1):
        # Gates
        masks = self.mask(t, s=s)
        gc1, gc2, gc3, gfc1, gfc2 = masks
        # Gated
        h = self.maxpool(self.drop1(self.relu(self.c1(x))))
        h = h*gc1.view(1, -1, 1, 1).expand_as(h)
        h = self.maxpool(self.drop1(self.relu(self.c2(h))))
        h = h*gc2.view(1, -1, 1, 1).expand_as(h)
        h = self.maxpool(self.drop2(self.relu(self.c3(h))))
        h = h*gc3.view(1, -1, 1, 1).expand_as(h)
        h = h.view(x.size(0), -1)
        h = self.drop2(self.relu(self.fc1(h)))
        h = h*gfc1.expand_as(h)
        h = self.drop2(self.relu(self.fc2(h)))
        h = h*gfc2.expand_as(h)
        y_hat = self.last[t](h)
        return y_hat, masks

    def mask(self, t, s=1):
        gc1 = self.gate(s*self.ec1(t))
        gc2 = self.gate(s*self.ec2(t))
        gc3 = self.gate(s*self.ec3(t))
        gfc1 = self.gate(s*self.efc1(t))
        gfc2 = self.gate(s*self.efc2(t))
        return [gc1, gc2, gc3, gfc1, gfc2]

    def get_view_for(self, n, masks):
        gc1, gc2, gc3, gfc1, gfc2 = masks
        if n == 'fc1.weight':
            post = gfc1.data.view(-1, 1).expand_as(self.fc1.weight)
            pre = gc3.data.view(-1, 1, 1).expand((self.ec3.weight.size(1), self.smid, self.smid)).contiguous().view(1, -1).expand_as(self.fc1.weight)
            return torch.min(post, pre)
        elif n == 'fc1.bias':
            return gfc1.data.view(-1)
        elif n == 'fc2.weight':
            post = gfc2.data.view(-1, 1).expand_as(self.fc2.weight)
            pre = gfc1.data.view(1, -1).expand_as(self.fc2.weight)
            return torch.min(post, pre)
        elif n == 'fc2.bias':
            return gfc2.data.view(-1)
        elif n == 'c1.weight':
            return gc1.data.view(-1, 1, 1, 1).expand_as(self.c1.weight)
        elif n == 'c1.bias':
            return gc1.data.view(-1)
        elif n == 'c2.weight':
            post = gc2.data.view(-1, 1, 1, 1).expand_as(self.c2.weight)
            pre = gc1.data.view(1, -1, 1, 1).expand_as(self.c2.weight)
            return torch.min(post, pre)
        elif n == 'c2.bias':
            return gc2.data.view(-1)
        elif n == 'c3.weight':
            post = gc3.data.view(-1, 1, 1, 1).expand_as(self.c3.weight)
            pre = gc2.data.view(1, -1, 1, 1).expand_as(self.c3.weight)
            return torch.min(post, pre)
        elif n == 'c3.bias':
            return gc3.data.view(-1)
        return None

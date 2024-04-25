import torch


class LightModel(torch.nn.Module):

    def __init__(self, alpha_leaky):
        super(LightModel, self).__init__()

        acti = torch.nn.LeakyReLU(alpha_leaky)

        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=3, padding='same')
        self.acti1 = acti
        self.conv2 = torch.nn.Conv2d(8, 4, kernel_size=3, padding='same')
        self.acti2 = acti
        self.conv3 = torch.nn.Conv2d(4, 2, kernel_size=9, padding='same', dilation=2)
        self.acti3 = acti
        self.conv4 = torch.nn.Conv2d(2, 1, kernel_size=1, padding='same')
        self.acti4 = acti

    def forward(self, x):
        x = self.conv1(x)
        x = self.acti1(x)
        x = self.conv2(x)
        x = self.acti2(x)
        x = self.conv3(x)
        x = self.acti3(x)
        x = self.conv4(x)
        x = self.acti4(x)
        return x
    
    def init_weights(self):
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.conv4.weight)

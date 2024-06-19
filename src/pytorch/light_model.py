import torch


class LightModel(torch.nn.Module):

    def __init__(self, alpha_leaky):
        super(LightModel, self).__init__()

        acti = torch.nn.LeakyReLU(alpha_leaky)

        self.drop1 = torch.nn.Dropout(p=0.30)
        self.conv1 = torch.nn.Conv2d( 3, 16, kernel_size=3, padding='same')
        self.acti1 = acti
        self.drop2 = torch.nn.Dropout(p=0.30)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding='same')
        self.acti2 = acti
        self.drop3 = torch.nn.Dropout(p=0.30)
        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=3, padding='same')
        self.acti3 = acti
        self.drop4 = torch.nn.Dropout(p=0.30)
        self.conv4 = torch.nn.Conv2d(32,  8, kernel_size=9, padding='same', dilation=2)
        self.acti4 = acti
        self.drop5 = torch.nn.Dropout(p=0.30)
        self.conv5 = torch.nn.Conv2d( 8,  4, kernel_size=1, padding='same')
        self.acti5 = acti
        self.drop6 = torch.nn.Dropout(p=0.15)
        self.conv6 = torch.nn.Conv2d( 4,  1, kernel_size=1, padding='same')

    def forward(self, x):
        x = self.drop1(x)
        x = self.conv1(x)
        x = self.acti1(x)
        x = self.drop2(x)
        x = self.conv2(x)
        x = self.acti2(x)
        x = self.drop3(x)
        x = self.conv3(x)
        x = self.acti3(x)
        x = self.drop4(x)
        x = self.conv4(x)
        x = self.acti4(x)
        x = self.drop5(x)
        x = self.conv5(x)
        x = self.acti5(x)
        x = self.drop6(x)
        x = self.conv6(x)
        # no sigmoid if we use BCEWithLogits
        return x
    
    def init_weights(self):
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.conv4.weight)

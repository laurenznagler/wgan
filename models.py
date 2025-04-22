import torch as th

class ExpLinearUnit(th.nn.Module):
    def __init__(self, alpha: float = 1.):
        super(ExpLinearUnit, self).__init__()
        self.alpha = th.nn.Parameter(th.tensor(alpha))
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        return th.where(x > 0, x, self.alpha * (th.exp(x) - 1.))
    
class ToyDiscriminator(th.nn.Module):
    def __init__(self):
        super(ToyDiscriminator, self).__init__()
        
        self.conv1 = th.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.conv2 = th.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.lin1 = th.nn.Linear(7 * 7 * 128, 1)
        self.act = ExpLinearUnit()
        
    def forward(self, x: th.Tensor):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.lin1(x.view(-1, 7 * 7 * 128))
        return x
    
class ToyGenerator(th.nn.Module):
    def __init__(self, input_size):
        super(ToyGenerator, self).__init__()
        self.input_size = input_size
        
        self.fc1 = th.nn.Linear(input_size, 5*5*256, bias = False)
        self.bn1 = th.nn.BatchNorm1d(5*5*256)
        
        self.conv1 = th.nn.ConvTranspose2d(256, 128, 4, bias = False, stride=1)
        self.bn2 = th.nn.BatchNorm2d(128)
        
        self.conv2 = th.nn.ConvTranspose2d(128, 64, 4, stride=2, bias = False, padding=2)
        self.bn3 = th.nn.BatchNorm2d(64)

        self.conv3 = th.nn.ConvTranspose2d(64, 1, 4, stride=2, bias = True, padding=1)
        
        self.act = ExpLinearUnit()
        
    def forward(self, x):        
        x = x.view(-1, self.input_size)
        x = self.act(self.bn1(self.fc1(x)))
        x = x.view(-1, 256, 5, 5)
        x = self.act(self.bn2(self.conv1(x)))   
        x = self.act(self.bn3(self.conv2(x)))     
        x = self.conv3(x)        
        return x
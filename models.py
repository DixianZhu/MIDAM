'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


def _weights_init(m):
    classname = m.__class__.__name__
    # print('initialize model')
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        # init.kaiming_normal_(m.weight)
        init.xavier_normal_(m.weight) 

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = activation_func(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = activation_func(out)
        return out



class MLP_stoc_MIL(nn.Module):
    def __init__(self, num_classes=1, last_activation='none', pretrained=False, dims=109):
        super(MLP_stoc_MIL, self).__init__()
        self.linear = nn.Linear(dims, num_classes)
        self.linear_1 = nn.Linear(dims, dims)
        self.attention = nn.Sequential(
            nn.Linear(dims, dims),
            nn.Tanh(),
            nn.Linear(dims, 1)
        )

        self.apply(_weights_init)
        print('model initialized')
        
        self.sigmoid = nn.Sigmoid()
        self.last_activation = last_activation
        self.bnlast = nn.BatchNorm1d(1)


    def forward(self, x):
        hidden = torch.tanh(self.linear_1(x))
        weights = self.attention(hidden)
        weights = torch.exp(weights)
        out = self.linear(hidden)
        return out, weights




class MLP_MIL(nn.Module):
    def __init__(self, num_classes=1, last_activation='none', pretrained=False, dims=109):
        super(MLP_MIL, self).__init__()
        self.linear = nn.Linear(dims, num_classes)
        self.linear_1 = nn.Linear(dims, dims)
        self.attention = nn.Sequential(
            nn.Linear(dims, dims),
            nn.Tanh(),
            nn.Linear(dims, 1)
        )

        self.apply(_weights_init)
        print('model initialized')
        
        self.sigmoid = nn.Sigmoid()
        self.last_activation = last_activation
        self.bnlast = nn.BatchNorm1d(1)


    def forward(self, x):
        hidden = torch.tanh(self.linear_1(x))
        hidden = hidden.view(hidden.shape[0],-1,hidden.shape[1])
        weights = torch.transpose(self.attention(hidden), 2, 1) # N x 1 x instance_num
        weights = F.softmax(weights, dim=2)
        hidden = torch.squeeze(torch.matmul(weights, hidden), dim=1)
        out = self.linear(hidden)
        if self.last_activation == 'sigmoid':
            out = self.sigmoid(out)
        elif self.last_activation == 'none' or self.last_activation==None:
            out = out#torch.clamp(out, min=-10, max=10)  
        elif self.last_activation == 'l2':
            out= F.normalize(out,dim=0,p=2)               
        elif self.last_activation == 'l1':
            out= F.normalize(out,dim=0,p=1)               
        elif self.last_activation == 'scale':
            out= (out - torch.min(out))/(torch.max(out)-torch.min(out)+1e-5)              
        elif self.last_activation == 'batchnorm':
            out= self.bnlast(out)            
        elif self.last_activation == 'sqrt':
            out= torch.sign(out)*torch.sqrt(torch.abs(out))      
        else:
            print('use the default sigmoid last activation!')
            out = self.sigmoid(out)
        
        return out



class MLP_softmax(nn.Module):
    def __init__(self, num_classes=1, last_activation='none', pretrained=False, dims=109, tau=1.0):
        super(MLP_softmax, self).__init__()
        self.linear = nn.Linear(dims, num_classes)
        self.linear_1 = nn.Linear(dims, dims)
        self.apply(_weights_init)
        print('model initialized')
        
        self.sigmoid = nn.Sigmoid()
        self.last_activation = last_activation
        self.bnlast = nn.BatchNorm1d(1)
        self.tau = tau


    def forward(self, x):
        hidden = torch.tanh(self.linear_1(x))
        out = self.linear(hidden)
        if self.last_activation == 'sigmoid':
            out = self.sigmoid(out)
        elif self.last_activation == 'none' or self.last_activation==None:
            out = out#torch.clamp(out, min=-10, max=10)  
        elif self.last_activation == 'l2':
            out= F.normalize(out,dim=0,p=2)               
        elif self.last_activation == 'l1':
            out= F.normalize(out,dim=0,p=1)               
        elif self.last_activation == 'scale':
            out= (out - torch.min(out))/(torch.max(out)-torch.min(out)+1e-5)              
        elif self.last_activation == 'batchnorm':
            out= self.bnlast(out)            
        elif self.last_activation == 'sqrt':
            out= torch.sign(out)*torch.sqrt(torch.abs(out))      
        else:
            print('use the default sigmoid last activation!')
            out = self.sigmoid(out)
        out = self.tau*torch.log(torch.mean(torch.exp(out/self.tau)))
        return out




class MLP(nn.Module):
    def __init__(self, num_classes=1, last_activation='none', pretrained=False, dims=109):
        super(MLP, self).__init__()
        self.linear = nn.Linear(dims, num_classes)
        self.linear_1 = nn.Linear(dims, dims)
        self.apply(_weights_init)
        print('model initialized')
        
        self.sigmoid = nn.Sigmoid()
        self.last_activation = last_activation
        self.bnlast = nn.BatchNorm1d(1)


    def forward(self, x):
        hidden = torch.tanh(self.linear_1(x))
        out = self.linear(hidden)
        if self.last_activation == 'sigmoid':
            out = self.sigmoid(out)
        elif self.last_activation == 'none' or self.last_activation==None:
            out = out#torch.clamp(out, min=-10, max=10)  
        elif self.last_activation == 'l2':
            out= F.normalize(out,dim=0,p=2)               
        elif self.last_activation == 'l1':
            out= F.normalize(out,dim=0,p=1)               
        elif self.last_activation == 'scale':
            out= (out - torch.min(out))/(torch.max(out)-torch.min(out)+1e-5)              
        elif self.last_activation == 'batchnorm':
            out= self.bnlast(out)            
        elif self.last_activation == 'sqrt':
            out= torch.sign(out)*torch.sqrt(torch.abs(out))      
        else:
            print('use the default sigmoid last activation!')
            out = self.sigmoid(out)
        return out


class ResNet_MIL(nn.Module):
    def __init__(self, block, num_blocks, inchannels=3, num_classes=1, last_activation='none'):
        super(ResNet_MIL, self).__init__()
        self.in_planes = 16
        self.inchannels = inchannels

        self.conv1 = nn.Conv2d(inchannels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.attention = nn.Sequential(
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)
        
        self.sigmoid = nn.Sigmoid()
        self.last_activation = last_activation
        self.bnlast = nn.BatchNorm1d(1)

    def init_weights(self):
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(-1,self.inchannels,x.shape[2],x.shape[3])
        out = activation_func(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = F.avg_pool2d(out, (out.size()[2],out.size()[3]))
        out = out.view(batch_size, -1, out.size()[1]) # N x instance_num x D
        weights = torch.transpose(self.attention(out), 2, 1) # N x 1 x instance_num
        weights = F.softmax(weights, dim=2)
        out = torch.squeeze(torch.matmul(weights, out), dim=1)
        hidden = out
        out = self.linear(out)
        if self.last_activation == 'sigmoid':
            out = self.sigmoid(out)
        elif self.last_activation == 'none' or self.last_activation==None:
            out = out 
        elif self.last_activation == 'l2':
            out= F.normalize(out,dim=0,p=2)               
        elif self.last_activation == 'l1':
            out= F.normalize(out,dim=0,p=1)               
        elif self.last_activation == 'scale':
            out= (out - torch.min(out))/(torch.max(out)-torch.min(out)+1e-5)              
        elif self.last_activation == 'batchnorm':
            out= self.bnlast(out)            
        elif self.last_activation == 'sqrt':
            out= torch.sign(out)*torch.sqrt(torch.abs(out))      
        else:
            print('use the default sigmoid last activation!')
            out = self.sigmoid(out)
        return out



class ResNet_stoc_MIL(nn.Module):
    def __init__(self, block, num_blocks, inchannels=3, num_classes=1, last_activation='none'):
        super(ResNet_stoc_MIL, self).__init__()
        self.in_planes = 16
        self.inchannels = inchannels

        self.conv1 = nn.Conv2d(inchannels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.attention = nn.Sequential(
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)
        
        self.sigmoid = nn.Sigmoid()
        self.last_activation = last_activation
        self.bnlast = nn.BatchNorm1d(1)

    def init_weights(self):
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(-1,self.inchannels,x.shape[2],x.shape[3])
        out = activation_func(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = F.avg_pool2d(out, (out.size()[2],out.size()[3]))
        out = out.view(out.size()[0], -1)
        weights = self.attention(out)
        weights = torch.exp(weights)
        out = self.linear(out)
        if self.last_activation == 'sigmoid':
            out = self.sigmoid(out)
        elif self.last_activation == 'none' or self.last_activation==None:
            out = out 
        elif self.last_activation == 'l2':
            out= F.normalize(out,dim=0,p=2)               
        elif self.last_activation == 'l1':
            out= F.normalize(out,dim=0,p=1)               
        elif self.last_activation == 'scale':
            out= (out - torch.min(out))/(torch.max(out)-torch.min(out)+1e-5)              
        elif self.last_activation == 'batchnorm':
            out= self.bnlast(out)            
        elif self.last_activation == 'sqrt':
            out= torch.sign(out)*torch.sqrt(torch.abs(out))      
        else:
            print('use the default sigmoid last activation!')
            out = self.sigmoid(out)
        return out, weights





class ResNet(nn.Module):
    def __init__(self, block, num_blocks, inchannels=3, num_classes=1, last_activation='none', pretrained=False):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(inchannels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.fc = nn.Linear(64, num_classes)

        self.apply(_weights_init)
        
        self.sigmoid = nn.Sigmoid()
        self.last_activation = last_activation
        self.bnlast = nn.BatchNorm1d(1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = activation_func(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # print(out.shape)
        # exit()
        
        out = F.avg_pool2d(out, (out.size()[2],out.size()[3]))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        if self.last_activation == 'sigmoid':
            out = self.sigmoid(out)
        elif self.last_activation == 'none' or self.last_activation==None:
            out = out#torch.clamp(out, min=-10, max=10)  
        elif self.last_activation == 'l2':
            out= F.normalize(out,dim=0,p=2)               
        elif self.last_activation == 'l1':
            out= F.normalize(out,dim=0,p=1)               
        elif self.last_activation == 'scale':
            out= (out - torch.min(out))/(torch.max(out)-torch.min(out)+1e-5)              
        elif self.last_activation == 'batchnorm':
            out= self.bnlast(out)            
        elif self.last_activation == 'sqrt':
            out= torch.sign(out)*torch.sqrt(torch.abs(out))      
        else:
            print('use the default sigmoid last activation!')
            out = self.sigmoid(out)
        return out

def ResNet20_stoc_MIL(pretrained=False, activations='relu', last_activation='sigmoid', **kwargs):
    global activation_func
    activation_func = F.relu if activations=='relu' else F.elu
    return ResNet_stoc_MIL(BasicBlock, [3, 3, 3], last_activation=last_activation, **kwargs)

def ResNet20_MIL(pretrained=False, activations='relu', last_activation='sigmoid', **kwargs):
    global activation_func
    activation_func = F.relu if activations=='relu' else F.elu
    return ResNet_MIL(BasicBlock, [3, 3, 3], last_activation=last_activation, **kwargs)

def ResNet20(pretrained=False, activations='relu', last_activation='sigmoid', **kwargs):
    global activation_func
    activation_func = F.relu if activations=='relu' else F.elu
    return ResNet(BasicBlock, [3, 3, 3], last_activation=last_activation, **kwargs)

def FFNN(pretrained=False, activations='relu', last_activation='sigmoid', **kwargs):
    global activation_func
    activation_func = F.relu if activations=='relu' else F.elu
    return MLP(last_activation=last_activation, **kwargs)

def FFNN_softmax(pretrained=False, activations='relu', last_activation='sigmoid', **kwargs):
    global activation_func
    activation_func = F.relu if activations=='relu' else F.elu
    return MLP_softmax(last_activation=last_activation, **kwargs)

def FFNN_MIL(pretrained=False, activations='relu', last_activation='sigmoid', **kwargs):
    global activation_func
    activation_func = F.relu if activations=='relu' else F.elu
    return MLP_MIL(last_activation=last_activation, **kwargs)

def FFNN_stoc_MIL(pretrained=False, activations='relu', last_activation='sigmoid', **kwargs):
    global activation_func
    activation_func = F.relu if activations=='relu' else F.elu
    return MLP_stoc_MIL(last_activation=last_activation, **kwargs)




def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()

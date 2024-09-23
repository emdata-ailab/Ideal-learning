import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet','resnet50']
 
model_urls = {
    # 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet50': '/workspace/hist_ideal_resnet50/code/net/resnet50-19c8e357.pth'
}


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class Embedding(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=None, normalized=True):
        super(Embedding, self).__init__()
        self.bn = nn.BatchNorm1d(out_dim, eps=1e-5)
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.linear.apply(weights_init_kaiming)
        print('out_dim is', out_dim)
        self.dropout = dropout
        self.normalized = normalized
        self.layernorm = nn.Sequential(
            nn.LayerNorm(out_dim, elementwise_affine = False)
        )       
        
        self.layernorm1 = nn.Sequential(
            nn.LayerNorm(in_dim, elementwise_affine = False)
        )       

    def forward(self, x, istest=False, normalized=True):
        
        if self.dropout is not None:
            x = nn.Dropout(p=self.dropout)(x, inplace=True)

        x = self.layernorm1(x)
        x = self.linear(x)  # x (batch_size, dim)
        
        if self.normalized and normalized:
            norm = x.norm(dim=1, p=2, keepdim=True)
            x = x.div(norm.expand_as(x))
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
 
 
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        identity = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out += identity
        out = self.relu(out)
 
        return out


class ResNet(nn.Module):
 
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, dim=512, head_type=''):
        super(ResNet, self).__init__()

        # add ideal network
        self.dim = dim
        self.multi_head = head_type != ''
        self.split = 'split' in head_type.lower()
        self.full_dim = 'full' in head_type.lower()
        # ========================================= #

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.num_ftrs = self.fc.in_features
        self.fc = None

        if self.dim == 0:
            pass
        elif not self.multi_head:
            self.classifier = Embedding(self.num_ftrs, self.dim, normalized=True)
        elif self.multi_head:
            input_dim = self.num_ftrs if not self.split else int(self.num_ftrs/4)
            dim = int(self.dim / 4) if not self.full_dim else self.dim
            self.classifier0 = Embedding(input_dim, dim, normalized=True)
            self.classifier1 = Embedding(input_dim, dim, normalized=True)
            self.classifier2 = Embedding(input_dim, dim, normalized=True)
            self.classifier3 = Embedding(input_dim, dim, normalized=True)
            
            self.classifier = [self.classifier0, self.classifier1, self.classifier2, self.classifier3]
        else:
            print("wrong params")
            raise

 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
 
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
 
        return nn.Sequential(*layers)
    

    def classify_forward(self, x, normalized=True):
        if not self.multi_head:
            x_out = self.classifier(x, normalized)
        else:
            if self.split:
                x_list = x.chunk(4,1)
                x_out = torch.cat([self.classifier[i](x_list[i], normalized) for i in range(4)], dim=1)
            else:
                x_out = torch.cat([self.classifier[i](x, normalized) for i in range(4)], dim=1)
            
        return x_out

 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 
        x = self.avgpool(x)

        # add ideal
        x = F.adaptive_max_pool2d(x, output_size=1)
        # ======================#

        x = x.view(x.size(0), -1)

        # add ideal
        if self.dim == 0:
            return x
        
        x = self.classify_forward(x, normalized=True)
        # ======================#

        # x = self.fc(x)
 
        return x
 
 
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        model_dict = model.state_dict() 
        pretrained_dict = torch.load(model_urls['resnet50']) 
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def main():
    model = resnet50(pretrained=True, normalized=True)
    print(model)
    images = torch.ones(8, 3, 227, 227)
    out_ = model(images)
    print(out_)

if __name__ == '__main__':
    main()

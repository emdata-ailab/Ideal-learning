# 将hist的结构集成到resnet和ideal的结构中
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['ResNet','resnet50']
 
# 这里定义了resnet的类型（18,34,50,101,152），以及各自的weigh文件的下载地址
model_urls = {
    # 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet50': '/workspace/hist_ideal_resnet50/code/net/resnet50-19c8e357.pth',
    'resnet_50': '/workspace/hist_ideal_resnet50/code/net/resnet50-0676ba61.pth',
}

def _init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class Embedding(nn.Module):
    def __init__(self, in_dim, embedding_size, is_norm, dropout_prob=0.5):
        super(Embedding, self).__init__()

        self.in_dim = in_dim
        self.embedding_size = embedding_size
        self.is_norm = is_norm

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.embedding = nn.Linear(self.in_dim, self.embedding_size)
        self._initialize_weights()
        
        if self.is_norm:
            self.layernorm = nn.LayerNorm(self.embedding_size, elementwise_affine=False)

    
    def ideal_norm(self, x):

        norm = x.norm(dim=1, p=2, keepdim=True)
        x = x.div(norm.expand_as(x))

        return x


    def forward(self, x):
        
        avg_x = self.gap(x)
        max_x = self.gmp(x)
        x = max_x + avg_x
       
        x = x.view(x.size(0), -1)

        x = self.embedding(x)

        if self.is_norm:
            x = self.layernorm(x)
        
        return x

    
    def _initialize_weights(self):
        init.kaiming_normal_(self.embedding.weight, mode='fan_out')
        init.constant_(self.embedding.bias, 0)

# 这里封装了两个Conv2d，第一个3*3：卷积核为3，padding为1，stride，in_channels和out_channels自定义。
# 第二个卷积核大小为1，没有padding。第一个3*3的主要作用是在以后高维中做卷积提取信息，第二个1*1的作用主要是进行升降维的。
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
 
 
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# 专门为深层结构设计的Block，其想法如下:首先通过一个1*1的卷积核对图像进行升维，然后通过3*3的卷积对其进行特征提取，
# 最后使用1*1的卷积进行降维等处理。由于有残差的存在和BasicBlock一样都有DownSample的操作。
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


# 首先看_make_layer函数，该函数是添加许多Block块至Sequential，
# 在该函数中首先定义：如果stride不等于1或者维度不匹配的时候的downsample，可以看到也是用过一个1*1的操作来进行升维的，
# 然后对其进行一次BN操作。接着对inplanes和planes不一致的情况进行了一次downsample ，即将带downsample的block添加至layers。
# 这样保证了x和out的维度一致，接下来通过一个循环添加了指定个数的Block，由于x已经维度一致了，这样添加的其他的Block就可以不用降维了，
# 所以循环添加不含Downsample的Block。
# 值得注意的是其中进行权重初始化，对Conv2d进行kaiming初始化，并且使用fan_out，保证了在反向传播中的数量参考这里。
# 对BN层进行权重为1，bias为0的初始化。还有一个参数就是在Bottleneck中的bn中最后一层对其进行0化。
class ResNet(nn.Module):
 
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True, embedding_size=512, is_norm=True, bn_freeze=True, head_type='', ):
        super(ResNet, self).__init__()

        # =========== add ideal network =========== #
        self.embedding_size = embedding_size
        self.multi_head = head_type != ''
        self.split = 'split' in head_type.lower()
        self.full_dim = 'full' in head_type.lower()
        self.is_norm = is_norm
        # ========================================= #

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # =========== add ideal network =========== #
        self.in_dim = self.fc.in_features
        # self.fc = None

        if not self.multi_head:
            self.classifier = Embedding(self.in_dim, self.embedding_size, is_norm)
        elif self.multi_head:
            input_dim = self.in_dim if not self.split else int(self.in_dim/4)
            output_dim = int(self.embedding_size / 4) if not self.full_dim else self.embedding_size
            self.classifier0 = Embedding(input_dim, output_dim, is_norm)
            self.classifier1 = Embedding(input_dim, output_dim, is_norm)
            self.classifier2 = Embedding(input_dim, output_dim, is_norm)
            self.classifier3 = Embedding(input_dim, output_dim, is_norm)
            
            self.classifier = [self.classifier0, self.classifier1, self.classifier2, self.classifier3]
        else:
            print("wrong params")
            raise
        # ========================================= #
 
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

        if bn_freeze:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)


 
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


    def classify_forward(self, x):
        if not self.multi_head:
            x_out = self.classifier(x)
        else:
            if self.split:
                x_list = x.chunk(4,1)
                x_out = torch.cat([self.classifier[i](x_list[i]) for i in range(4)], dim=1)
            else:
                x_out = torch.cat([self.classifier[i](x) for i in range(4)], dim=1)
            
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
        
        x = self.classify_forward(x)
 
        return x
 
 
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_urls['resnet50']) 
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def main():
    model = resnet50(pretrained=True)
    print(model)
    images = torch.ones(8, 3, 227, 227)
    out_ = model(images)
    print(out_)

if __name__ == '__main__':
    main()

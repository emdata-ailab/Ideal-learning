import torch
from torch.autograd import Variable
def tensor_hook(grad):
    print('tensor hook')
    print('grad:', grad)
    return grad

class test:
    def __init__(self):
        # print('type a', type(self.a[0]))
        with torch.no_grad():
            x = torch.tensor([1], dtype=torch.float32, requires_grad=True)
            x.retain_grad()
            y = x*2
           
            self.a = y
            self.b = self.a
        self.b.retain_grad()
        print('self.b.requires_grad', self.b.requires_grad)

    def get(self, t):
        print('self.b.requires_grad in test',self.b[0].requires_grad)
        self.b.data[0] = t
        print('t.requires_grad in test', t.requires_grad)
        print('self.b.requires_grad in test',self.b[0].requires_grad)
        return self.b
    
    def print_grad(self):
        print('inside:', self.b.grad)

x = torch.tensor([1], dtype=torch.float32, requires_grad=True)
x.retain_grad()
y = x*2
print('y', y.requires_grad)
b = test()
m = b.get(y)
m.retain_grad()
z = torch.sum(m)
# print('y.requires_grad', y.requires_grad)
# print('t.requires_grad', t.requires_grad)
z.backward()
b.print_grad()
print(m.grad)


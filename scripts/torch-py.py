import torch;

t1 = torch.zeros((100, 100, 100))
t2 = torch.ones((100, ))

t3 = t1 + t2

print(f'{t1.shape}')
print(f'{t2.shape}')


print(f'{t3.shape}')


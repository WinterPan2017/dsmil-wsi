import torch
# x= torch.randn((10,1), requires_grad=True)

# t, _ = torch.max(x, dim=0, keepdim=True)
# f = torch.nn.functional.binary_cross_entropy_with_logits(t, torch.ones((1,1)))
# f.backward()
# print(torch.argmax(x, dim=0))
# print(torch.argmin(x.grad, dim=0))
# print(x.grad)

x = torch.randn((10, 1), requires_grad=True)
indices = torch.tensor([2])
t = torch.index_select(x, 0, indices)
f = torch.nn.functional.binary_cross_entropy_with_logits(t, torch.ones((1,1)))
f.backward()
print(x.grad)



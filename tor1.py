import torch
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.randn(1, 3, 64, 64)
labels = torch.rand(1, 1000)

# Forward pass
prediction = model(data)

loss = (prediction - labels).sum()
loss.backward()

optim = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)

optim.step()


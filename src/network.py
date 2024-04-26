import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch import optim
from reformat_images import generate_dataset
from PIL import Image
import os



transform = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])





# Define classifier model
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        # Add more layers as needed
        num_classes = 4
        self.fc = nn.Linear(64 * 244 * 244, num_classes)  # Adjust input size based on conv layers

    def forward(self, x):
        x = self.conv1(x)
        # Apply more layers and activations as needed
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Initialize model and optimizer

data = generate_dataset()
model = Classifier()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
num_epochs = 100
# Training loop (pseudo code)
for epoch in range(num_epochs):
    model.train()
    for images, labels in data:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # Print or log training loss

# Evaluation loop (pseudo code)
model.eval()
total_correct = 0
total_samples = 0



for images, labels in data:
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    total_samples += labels.size(0)
    total_correct += (predicted == labels).sum().item()

accuracy = total_correct / total_samples
print("Accuracy:", accuracy)

import matplotlib.pyplot as plt
import torchvision
import torch
from torchvision.datasets import CIFAR10
import torch.nn as nn
import torch.optim as optim

# Data PreProcessing 
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
 
batch_size = 16

#Using Pytorch dataloader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

# Visualizing CIFAR10
def imshow(dataloader,batchsize=8):
    fig, ax = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(12,8))
    for images, labels in trainloader:
        for i in range(batch_size):
            row, col = i//4, i%4
            ax[row][col].imshow(images[i].numpy().transpose([1,2,0]))
        break  # take only the first batch
    plt.show()
    
# imshow(trainloader,batchsize=batch_size)

# Convolutional Neural Network
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2)),
    nn.Flatten(),
    nn.Linear(8192, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 10)
)
print(model)

# torch.save(model.state_dict(), 'MLMAstery/cifar10_model.pt')




# Train an Image Classifier using GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
 
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
 
n_epochs = 15
def Training(epochs,loss_fn,optimizer):
    for epoch in range(epochs):
        model.train()
        for inputs, labels in trainloader:
            y_pred = model(inputs.to(device))
            loss = loss_fn(y_pred, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc = 0
        count = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                y_pred = model(inputs.to(device))
                acc += (torch.argmax(y_pred, 1) == labels.to(device)).float().sum()
                count += len(labels)
        acc /= count
        print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))

Training(epochs=n_epochs,loss_fn=loss_fn,optimizer=optimizer)
        
   
    
# testing on test dataset
def TestingOnTestdata(model, testloader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

        print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))

saved = torch.load('cifar10_model.pt')
model.load_state_dict(saved)
TestingOnTestdata(model=model, testloader=testloader)
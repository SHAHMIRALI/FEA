import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
# import sklearn as sk
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm

#Step 1 bring in your data and augment it
#DATA AUGMENTATION

train_transform = transforms.Compose([transforms.RandomVerticalFlip(p=0.5),
                                transforms.RandomHorizontalFlip(p=0.5),
                                #transforms.RandomResizedCrop(28),
                                transforms.ColorJitter(hue=.05, saturation=.05),
                                transforms.RandomRotation(30),
                                transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5)])

test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(0.5, 0.5)])

# Define a transform to normalize the data

#APPLY TRANSFORMATION TO TRAINING AND TEST SET
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


#GET NEXT IMAGE AND LABEL FROM CURRENT BATCH (training and test batch)
image, label = next(iter(trainloader))
plt.imshow(image[0][0], cmap='gray')
plt.title(trainset.classes[label[0].numpy()])
plt.axis('off')
plt.show()

image, label = next(iter(testloader))
plt.imshow(image[0][0], cmap='gray')
plt.title(testset.classes[label[0].numpy()])
plt.axis('off')
plt.show()

# Step 2 define your neural network (2 examples below)
# TODO: Define your network architecture here
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        #define # of inputs and # of outputs of each layer
        #each image is 28x28 (784) and final layer has output = # of labels(10)
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)

# Step 3 Train your model (using CNN model for this example)
# TODO: Create the network, define the criterion and optimizer
convolution = True

if convolution:
    model = CNN()

else:
    model = Classifier()

#create loss function
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

#the higher the better
epochs = 5

model.train()

for e in range(epochs):
    running_loss = 0
    #for each batch of images put them through the model
    for images, labels in tqdm(trainloader):
        log_ps = model(images)
        loss = criterion(log_ps, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss / len(trainloader)}")

# Step 4 Test your model
dataiter = iter(testloader)
images, labels = dataiter.next()
#Change the index of image in batch to visualize its confidence score
index = 63
img = images[index]
img_to_show=img[0]

if convolution:
  img = img.unsqueeze(0)

ps = torch.exp(model(img))

plt.figure(figsize=(15, 5))

plt.subplot(121)
plt.imshow(img_to_show, cmap='gray')
plt.title('Correct label: '+testset.classes[labels[index].numpy()])
plt.axis('off')

plt.subplot(122)
plt.barh(np.arange(len(testset.classes)), ps.detach().numpy()[0])
plt.yticks(ticks=np.arange(len(testset.classes)), labels=testset.classes)
plt.title('Model\'s predicted probability vector')
plt.show()


#Qualitative testing (test all images against the model in a loop)
model.eval()

accuracy = 0
for images, labels in testloader:
    log_ps = model(images)

    # get the predictions: argmax etc.
    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy += torch.mean(equals.type(torch.FloatTensor))

print("Test Accuracy: {:.3f}".format(accuracy / len(testloader)))

#Step 5 using a validation set for training (improves accuracy) optional
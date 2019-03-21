import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import fruit_data
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import accuracy_score

device = torch.device("cuda") if (torch.cuda.is_available()) else torch.device("cpu")
print (device)

class FruitNet(nn.Module):
    def __init__(self):
        super(FruitNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64,64, kernel_size=7, stride=1)
        self.pool2 = nn.MaxPool2d(3)
        self.conv3 = nn.Conv2d(64,64, kernel_size=7)
        self.pool3 = nn.MaxPool2d(5)
        self.linear1 = nn.Linear(64, 100)
        self.linear2 = nn.Linear(100, 65) 

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        return x

def train_network(dataloader_train):
    net = FruitNet()
    net = net.to(device)
    #optimizer = torch.optim.Adagrad(net.parameters(), lr=0.01)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 10
    losses = []
    for epoch in range(epochs):
        current_loss = 0.0
        print ("Epoch : {}".format(epoch + 1))
        for i_batch, (images, labels) in enumerate(dataloader_train):
            images, labels = images.to(device), labels.to(device)
            x = Variable(images, requires_grad=False).float()
            y = Variable(labels, requires_grad=False).long()
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = net(x)
            correct = y_pred.max(1)[1].eq(y).sum()
            print ("INFO: Number of correct items classified : {}".format(correct.item()))
            loss = criterion(y_pred, y)
            print ("Loss : {}".format(loss.item()))
            current_loss += loss.item()
            loss.backward()
            optimizer.step()
        losses.append(current_loss)

    ## Save the network.
    torch.save(net.state_dict(), "model/fruit_model_state_dict.pth")
    torch.save(optimizer.state_dict(), "model/fruit_model_optimizer_dict.pth")
    print ("OK: Finished training for {} epochs".format(epochs))

    return losses, net

def test_network(net, dataloader_test):
    net.eval()
    criterion = nn.CrossEntropyLoss()
    accuracies = []
    with torch.no_grad():
        for feature, label in dataloader_test:
            feature = feature.to(device)
            label = label.to(device)
            pred = net(feature)
            accuracy = accuracy_score(label.cpu().data.numpy(), pred.max(1)[1].cpu().data.numpy()) * 100
            print ("Accuracy : ", accuracy)
            loss = criterion(pred, label)
            print ("Loss : {}".format(loss.item()))
            accuracies.append(accuracy)
    
    total = 0.0
    for j in range(len(accuracies)):
        total = total + accuracies[j]
    avg_acc = total / len(accuracies)
    print ("OK: testing done with overall accuracy is : {}".format(avg_acc))
        
def main():
    root_dir = args.data_dir
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    transformed_dataset = fruit_data.Fruit(root_dir, train=True, transform=data_transform)
    dataloader_train = DataLoader(transformed_dataset, batch_size=16, shuffle=True, num_workers=4)
    transformed_test_dataset = fruit_data.Fruit(root_dir, train=False, transform=data_transform)
    dataloader_test = DataLoader(transformed_test_dataset, batch_size=16, shuffle=True, num_workers=4)
    
    dataiter = iter(dataloader_train)
    images, labels = dataiter.next()
    print ("INFO: image shape is {}".format(images.shape))
    print ("INFO: Tensor type is : {}".format(images.type()))
    print ("INFO: labels shape is : {}".format(labels.shape))

    losses, net = train_network(dataloader_train)
    test_network (net, dataloader_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help="Dataset directory where npy files are stored")
    args = parser.parse_args()
    main()

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.transforms import v2


# ============================================================
# 模型
# ============================================================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 62)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ============================================================
# 資料
# ============================================================
def get_transforms():
    """回傳 (train_tf, test_tf)"""
    IMG_SIZE = 28

    train_tf = v2.Compose([
        v2.ToImage(),
        v2.Grayscale(1),
        v2.Resize((IMG_SIZE, IMG_SIZE)),
        v2.RandomCrop(IMG_SIZE, padding=2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5,), (0.5,)),
    ])

    test_tf = v2.Compose([
        v2.ToImage(),
        v2.Grayscale(1),
        v2.Resize((IMG_SIZE, IMG_SIZE)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5,), (0.5,)),
    ])

    return train_tf, test_tf


def get_dataloaders():
    """回傳 (train_loader, testloader, classes)"""
    train_dir = "./data/handwriting/augmented_images/augmented_images1"
    test_dir  = "./data/handwriting/handwritten-english-characters-and-digits/combined_folder/test"
    batch_size = 32

    train_tf, test_tf = get_transforms()

    trainset = torchvision.datasets.ImageFolder(train_dir, transform=train_tf)
    testset  = torchvision.datasets.ImageFolder(test_dir,  transform=test_tf)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=10, pin_memory=torch.cuda.is_available())
    testloader   = torch.utils.data.DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=torch.cuda.is_available())

    return train_loader, testloader, testset.classes


# ============================================================
# 訓練與評估
# ============================================================
def train(net, train_loader, criterion, optimizer, device):
    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
                running_loss = 0.0

    print('Finished Training')


def test(net, testloader, criterion, device):
    """計算整體測試 loss 與正確率"""
    net.eval()

    correct = 0
    total = 0
    test_loss = 0.0

    print("開始評估...")

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # 計算損失
    average_test_loss = test_loss / len(testloader.dataset)
    accuracy = 100. * correct / total

    print(f'\n[Test Result] loss: {average_test_loss:.3f} | Accuracy: {accuracy:.2f}% ({correct}/{total})')
    print('Finished Testing')


def evaluate(net, testloader, classes, device):
    """統計每個類別的正確率"""
    class_correct = list(0. for _ in range(62))
    class_total = list(0. for _ in range(62))

    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(len(classes)):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f'類別 {classes[i]:<5s} 的正確率 : {acc:.1f}% ({int(class_correct[i])}/{int(class_total[i])})')
        else:
            print(f'類別 {classes[i]:<5s} 沒有測試樣本')


# ============================================================
# 模型存取
# ============================================================
def save_model(net, model_path):
    """儲存模型權重"""
    torch.save(net.state_dict(), model_path)
    print(f'模型已儲存: {model_path}')


def load_model(net, model_path, device):
    """載入模型權重"""
    net.load_state_dict(torch.load(model_path, map_location=device))
    print(f'已載入模型: {model_path}')


# ============================================================
# 主程式
# ============================================================
def main():
    train_loader, testloader, classes = get_dataloaders()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用裝置: {device}")

    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # 已經有存好的模型就直接載入，沒有才訓練並存檔
    model_path = './handwriting_cnn.pth'
    if os.path.exists(model_path):
        load_model(net, model_path, device)
    else:
        train(net, train_loader, criterion, optimizer, device)
        save_model(net, model_path)

    test(net, testloader, criterion, device)
    evaluate(net, testloader, classes, device)


if __name__ == '__main__':
    main()

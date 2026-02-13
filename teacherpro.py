import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

# ================= 配置区域 =================
# 这里的路径改成你实际的路径
DATA_PATH = r'D:\数据库\COVID-19_Radiography_Dataset\a'

# 图像尺寸 128x128
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 60  # 老师模型收敛快，40轮足够
NUM_CLASSES = 4


def train_teacher():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Teacher using device: {device}")

    # 1. 数据准备
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        # 老师可以用归一化，因为学生会模仿老师的输出分布，不需要输入分布完全一致
        # 但为了稳妥，这里保持和学生一致，不做 mean/std 归一化
    ])

    try:
        full_dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
        # 8:2 划分
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print("Data Loaded Successfully.")
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. 定义老师模型 (ResNet50)
    # 使用预训练权重，训练更快
    print("Loading ResNet50...")
    model = models.resnet50(pretrained=True)

    # 修改全连接层以匹配我们的 4 分类
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # 老师可以用稍小的学习率微调
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_acc = 0.0
    save_path = 'resnet50_teacher_best.pth'

    # 3. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        print(f"Teacher Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            print(f"  >>> New Best Teacher Saved! (Acc: {best_acc:.2f}%)")

    print(f"Teacher training finished. Best path: {save_path}")


if __name__ == '__main__':
    train_teacher()
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
import numpy as np

# ==========================================
# 1. 配置区域
# ==========================================
# 数据集路径
DATA_PATH = r'D:\数据库\COVID-19_Radiography_Dataset\a'
import torch.nn.functional as F  # 蒸馏需要用到
from torch.utils.data import DataLoader, random_split
# 老师模型路径 (必须先运行第一步代码生成这个文件)
TEACHER_PATH = 'resnet50_teacher_best.pth'

CHANNEL_SCALE = 1.0
EPOCHS = 120  # 蒸馏需要较多轮次来吸收知识

# 蒸馏超参数
TEMPERATURE = 7.0  # 温度：越高，老师输出的概率分布越平滑，包含的暗知识越多
ALPHA = 0.7  # 权重：0.7表示 70% 听老师的，30% 看标准答案

# 定点化参数 (用于导出)
TOTAL_BITS = 16
INT_BITS = 8
FRAC_BITS = TOTAL_BITS - INT_BITS
SCALE_FACTOR = 2 ** FRAC_BITS


def get_ch(base_ch):
    return int(base_ch * CHANNEL_SCALE)


# ==========================================
# 2. 学生模型 (FPGA_Standard_CNN) - 保持硬件兼容性
# ==========================================
class FPGA_Standard_CNN(nn.Module):
    def __init__(self, num_classes=4):
        super(FPGA_Standard_CNN, self).__init__()

        # Stride=1 + MaxPool 结构，节省 BRAM/DSP 且精度高
        self.conv1 = nn.Conv2d(3, get_ch(16), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(get_ch(16))
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(get_ch(16), get_ch(32), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(get_ch(32))

        self.conv3 = nn.Conv2d(get_ch(32), get_ch(64), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(get_ch(64))

        self.conv4 = nn.Conv2d(get_ch(64), get_ch(128), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(get_ch(128))

        fc_in_features = 8 * 8 * get_ch(128)
        self.fc = nn.Linear(fc_in_features, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ==========================================
# 3. 蒸馏损失函数
# ==========================================
def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    # 1. 软目标损失 (向老师学习)
    # KLDivLoss 要求输入是 LogSoftmax，目标是 Softmax
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1)
    ) * (T * T)

    # 2. 硬目标损失 (向标准答案学习)
    hard_loss = F.cross_entropy(student_logits, labels)

    # 3. 加权求和
    return alpha * soft_loss + (1.0 - alpha) * hard_loss


# ==========================================
# 4. 导出工具 (BN融合 + 量化)
# ==========================================
def fuse_bn_and_quantize(conv, bn):
    with torch.no_grad():
        w = conv.weight.clone()
        mean = bn.running_mean
        var_sqrt = torch.sqrt(bn.running_var + bn.eps)
        gamma = bn.weight
        beta = bn.bias
        scale = gamma / var_sqrt
        w_fused = w * scale.reshape(-1, 1, 1, 1)
        b_fused = (0 - mean) * scale + beta
        w_int = torch.round(w_fused * SCALE_FACTOR).clamp(-32768, 32767).numpy().astype(np.int16)
        b_int = torch.round(b_fused * SCALE_FACTOR).clamp(-32768, 32767).numpy().astype(np.int16)
        return w_int, b_int


def quantize_fc(fc_layer):
    with torch.no_grad():
        w = fc_layer.weight.clone()
        b = fc_layer.bias.clone()
        w_int = torch.round(w * SCALE_FACTOR).clamp(-32768, 32767).numpy().astype(np.int16)
        b_int = torch.round(b * SCALE_FACTOR).clamp(-32768, 32767).numpy().astype(np.int16)
        return w_int, b_int


def export_weights_to_h(model, filename="weights.h"):
    print(f"Generating {filename} with Distilled Knowledge...")
    with open(filename, 'w') as f:
        f.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n#include <ap_fixed.h>\n\n")

        layers = [(model.conv1, model.bn1, "conv1"), (model.conv2, model.bn2, "conv2"),
                  (model.conv3, model.bn3, "conv3"), (model.conv4, model.bn4, "conv4")]

        for conv, bn, name in layers:
            w_int, b_int = fuse_bn_and_quantize(conv, bn)
            f.write(
                f"const short {name}_bias[{b_int.shape[0]}] = {{\n" + ", ".join(map(str, b_int.flatten())) + "\n};\n\n")
            f.write(f"const short {name}_weights[{w_int.shape[0]}][{w_int.shape[1]}][3][3] = {{\n")
            for o in range(w_int.shape[0]):
                f.write("  {\n")
                for i in range(w_int.shape[1]):
                    f.write("    {" + ", ".join(map(str, w_int[o, i].flatten())) + "},\n")
                f.write("  },\n")
            f.write("};\n\n")

        w_fc, b_fc = quantize_fc(model.fc)
        f.write(f"const short fc_bias[{b_fc.shape[0]}] = {{\n" + ", ".join(map(str, b_fc.flatten())) + "\n};\n\n")
        f.write(f"const short fc_weights[{w_fc.size}] = {{\n" + ", ".join(map(str, w_fc.flatten())) + "\n};\n\n")
        f.write("#endif\n")
    print("Done! weights.h generated.")


# ==========================================
# 5. 主训练流程
# ==========================================
def train_distillation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Distillation using device: {device}")

    # --- 1. 数据 ---
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    try:
        full_dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    except Exception as e:
        print(f"Error loading data: {e}");
        return

    # --- 2. 加载老师 ---
    print("Loading Teacher (ResNet50)...")
    teacher = models.resnet50(pretrained=False)
    teacher.fc = nn.Linear(teacher.fc.in_features, 4)
    if os.path.exists(TEACHER_PATH):
        teacher.load_state_dict(torch.load(TEACHER_PATH))
    else:
        print("Error: Teacher weights not found! Run train_teacher_pro.py first.")
        return
    teacher.to(device)
    teacher.eval()  # 老师必须冻结

    # --- 3. 初始化学生 ---
    print("Initializing Student (FPGA_Standard_CNN)...")
    student = FPGA_Standard_CNN(num_classes=4).to(device)
    optimizer = optim.Adam(student.parameters(), lr=0.001)

    best_acc = 0.0
    save_path = 'best_student_distilled.pth'

    print("Start Distillation Training...")
    for epoch in range(EPOCHS):
        student.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # 前向传播
            student_logits = student(images)
            with torch.no_grad():
                teacher_logits = teacher(images)

            # 计算蒸馏损失
            loss = distillation_loss(student_logits, teacher_logits, labels, TEMPERATURE, ALPHA)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 验证
        student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = student(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {running_loss / len(train_loader):.4f} | Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), save_path)
            print("  >>> Best Student Saved!")

    print(f"\nTraining Finished. Best Acc: {best_acc:.2f}%")

    # --- 4. 导出 ---
    student.load_state_dict(torch.load(save_path))
    student.cpu().eval()
    export_weights_to_h(student)


if __name__ == '__main__':
    train_distillation()
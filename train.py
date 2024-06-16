import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from net import vgg16  # 确保你的 net.py 中有 vgg16 定义
from data import DataGenerator  # 确保你的 data.py 中有 DataGenerator 定义
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载被截断的图像

def main():
    # Dataset
    annotation_path = 'cls_honda_cars.txt'
    with open(annotation_path, 'r') as f:
        lines = f.readlines()

    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val

    # Input image size
    input_shape = [224, 224]

    train_data = DataGenerator(lines[:num_train], input_shape, True)
    val_data = DataGenerator(lines[num_train:], input_shape, False)
    val_len = len(val_data)
    print(f'Validation data length: {val_len}')  # 验证集长度

    # Increase batch size and number of workers
    batch_size = 128  # Adjust this based on your GPU memory
    num_workers = 4  # Increase the number of workers to load data faster

    # DataLoader
    gen_train = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    gen_test = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Network construction
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = vgg16(pretrained=True, progress=True, num_classes=17)
    net.to(device)

    # Optimizer and learning rate scheduler
    lr = 0.0001
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1)

    # Training
    epochs = 10
    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}")
        total_train_loss = 0
        net.train()
        for data in gen_train:
            img, label = data
            img = img.to(device)
            label = label.to(device)

            optim.zero_grad()
            output = net(img)
            train_loss = nn.CrossEntropyLoss()(output, label)
            train_loss.backward()
            optim.step()
            total_train_loss += train_loss.item()

        scheduler.step()

        total_test_loss = 0
        total_accuracy = 0
        net.eval()
        for data in gen_test:
            img, label = data
            img = img.to(device)
            label = label.to(device)

            with torch.no_grad():
                output = net(img)
                test_loss = nn.CrossEntropyLoss()(output, label)
                total_test_loss += test_loss.item()
                accuracy = (output.argmax(1) == label).sum().item()
                total_accuracy += accuracy

        avg_train_loss = total_train_loss / len(gen_train)
        avg_test_loss = total_test_loss / len(gen_test)
        avg_accuracy = total_accuracy / val_len

        print("Epoch {}: Train Loss: {:.4f}, Test Loss: {:.4f}, Accuracy: {:.2%}".format(
            epoch + 1, avg_train_loss, avg_test_loss, avg_accuracy))

        torch.save(net.state_dict(), "Car_{}.pth".format(epoch + 1))
        print(f"Model saved for epoch {epoch + 1}")

if __name__ == '__main__':
    main()

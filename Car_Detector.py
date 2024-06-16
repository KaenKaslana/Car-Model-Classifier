import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from net import vgg16  # 确保你的 net.py 中有 vgg16 定义

# 定义模型
def load_model(model_path, num_classes):
    model = vgg16(pretrained=False, progress=True, num_classes=num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# 预处理输入图像
def preprocess_image(image_path, input_shape):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(input_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # 增加批次维度
    return image

# 进行预测
def predict(model, device, image_path, input_shape):
    image = preprocess_image(image_path, input_shape)
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
    prediction = torch.argmax(output, dim=1).item()
    return prediction

if __name__ == '__main__':
    # 模型路径
    model_path = 'D:\CarDetector\Trained_Parameters\Car_10.pth'
    input_shape = [224, 224]  # 图像输入尺寸
    num_classes = 17  # 分类的类别数目

    # 类别名称
    class_names = ['accord', 'amaze', 'brio', 'city', 'civic', 'clarity', 'freed', 'insight', 'legend', 'mobilo', 
                   'nsx', 'odyssey', 'passport', 'pilot', 'ridgeline', 's660', 'vezel']

    # 加载模型
    model, device = load_model(model_path, num_classes)

    # 预测示例图像路径
    image_path = 'D:\CarDetector\Put_your_Image_here\image.jpg'  # 替换为你要预测的图像路径

    # 进行预测
    prediction = predict(model, device, image_path, input_shape)
    
    # 输出预测结果
    print(f'The predicted class is: {class_names[prediction]}')

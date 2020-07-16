import torch
import cv2
import torch.nn.functional as F
from cifar import Net
from torchvision import datasets, transforms
import numpy as np

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('model.pth')
    model = model.to(device)
    model.eval()

    img = cv2.imread(r'C:\Users\Lenovo\Desktop\cat2.jpg')
    cv2.imshow('image', img)
    img = cv2.resize(img, dsize=(32, 32))
    print(img.shape)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    img = trans(img)
    img = img.to(device)
    img = img.unsqueeze(0)

    output = model(img)
    prob = F.softmax(output, dim=1)
    print(prob)
    value, predicted = torch.max(output.data, 1)
    print(predicted.item())
    print(value)
    pred_class = classes[predicted.item()]
    print(pred_class)
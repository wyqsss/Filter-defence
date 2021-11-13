import argparse
from kornia import filters
import torch
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
from model import MobileNetV1
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
import kornia


def PGD(model, test_loader, filter=None):
    adver_example = None
    adver_target = None
    correct = 0.0
    for (data, target) in test_loader:
        data = Variable(data)
        target = Variable(target)
        data = data.cuda()
        target = target.cuda()
        # model_fn = lambda x:F.nll_loss(model(x),target.to(device))
        adver_example = projected_gradient_descent(
            model, data, 0.1, 0.05, 40, np.inf)
        if filter == "blur":
            adver_example = kornia.filters.box_blur(
                adver_example, (3, 3))                 # 使用均值滤波防御
        elif filter == "median":
            adver_example = kornia.filters.median_blur(
                adver_example, (3, 3))              # 使用中值滤波防御
        elif filter == "gaussian":
            adver_example = kornia.filters.gaussian_blur2d(
                adver_example, (3, 3), (1, 1))  # 使用高斯滤波防御
        adver_target = torch.max(model(adver_example), 1)[1]
        # print(f"adver_targer is : {adver_target}")
        correct += adver_target.eq(target).sum()  # 统计相等的个数
    if not filter:
        print(
            f"acc after PGD attack: {correct.float()/len(test_loader.dataset)}")
    else:
        print(
            f"acc defence by {filter} after PGD attack: {correct.float()/len(test_loader.dataset)}")
    return correct.float()/len(test_loader.dataset)


def FGSM(model, test_loader, filter=None):
    adver_example = None
    adver_target = None
    correct = 0.0
    for (data, target) in test_loader:
        data = Variable(data)
        target = Variable(target)
        data = data.cuda()
        target = target.cuda()
        # model_fn = lambda x:F.nll_loss(model(x),target.to(device))
        adver_example = fast_gradient_method(model, data, 0.1, np.inf)
        if filter == "blur":
            adver_example = kornia.filters.box_blur(
                adver_example, (3, 3))                 # 使用均值滤波防御
        elif filter == "median":
            adver_example = kornia.filters.median_blur(
                adver_example, (3, 3))              # 使用中值滤波防御
        elif filter == "gaussian":
            adver_example = kornia.filters.gaussian_blur2d(
                adver_example, (3, 3), (1, 1))  # 使用高斯滤波防御

        adver_target = torch.max(model(adver_example), 1)[1]
        # print(f"adver_targer is : {adver_target}")
        correct += adver_target.eq(target).sum()  # 统计相等的个数
    if not filter:
        print(
            f"acc after FGSM attack: {correct.float()/len(test_loader.dataset)}")
    else:
        print(
            f"acc defence by {filter} after FGSM attack: {correct.float()/len(test_loader.dataset)}")
    return correct.float()/len(test_loader.dataset)


def test(net, test_loader):
    net.eval()
    correct = 0.0
    for (images, labels) in test_loader:
        images = Variable(images)
        labels = Variable(labels)
        # print(labels)
        # print(type(labels))
        images = images.cuda()
        labels = labels.cuda()
        images = images
        outputs = net(images)
        _, preds = outputs.max(1)  # 最大值的index
        correct += preds.eq(labels).sum()  # 统计相等的个数

    print('Test set:  Accuracy: {:.4f}'.format(
        correct.float() / len(test_loader.dataset)
    ))

    return correct.float() / len(test_loader.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='data path')
    parser.add_argument('--weights', type=str, default="",
                        help='the weights file you want to test')
    parser.add_argument('--attack', type=str, default="", help='选择攻击类型')
    args = parser.parse_args()
    model = MobileNetV1()
    model.load_state_dict(torch.load(
        args.weights, map_location=torch.device('cpu')))
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    testset = torchvision.datasets.CIFAR10(
        root=args.data, train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2)
    model = model.cuda()
    acc = test(model, test_loader)
    if args.attack == "fgsm":
        FGSM(model, test_loader)             # 进行攻击
        FGSM(model, test_loader, "blur")     # 使用均值滤波进行防御
        FGSM(model, test_loader, "median")   # 使用中值滤波进行防御
        FGSM(model, test_loader, "gaussian")  # 使用高斯滤波进行防御
    elif args.attack == "pgd":
        PGD(model, test_loader)             # 进行攻击
        PGD(model, test_loader, "blur")     # 使用均值滤波进行防御
        PGD(model, test_loader, "median")   # 使用中值滤波进行防御
        PGD(model, test_loader, "gaussian")  # 使用高斯滤波进行防御

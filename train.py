from models.resnet.model import resnet101
from torchsummary import summary





def main():
    model = resnet101(3, 1000)
    summary(model, (3, 224, 224))


if __name__ == '__main__':
    main()

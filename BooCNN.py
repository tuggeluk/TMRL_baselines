import torch.nn.functional as F
import torchvision.transforms
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
from torchvision.transforms import functional as F_tv
from datasets import load_dataset


class BooCNN(nn.Module):
    def __init__(self, in_channels):
        super(BooCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=11, stride=1, padding=1)
        self.lrn1 = nn.LocalResponseNorm(2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2, padding=1)
        self.template = nn.Conv2d(in_channels=256, out_channels=1,kernel_size=12)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.template(x)
        x = self.sigm(x)
        return torch.squeeze(x)


class ImgNetCollator:
    def __init__(self, p_rot = 0.33):
        self.res = torchvision.transforms.Resize((244,244))
        self.rr = torchvision.transforms.RandomRotation(degrees=(1, 259))
        x = np.arange(0, 244, 1) - np.floor(244 / 2)
        y = np.arange(0, 244, 1) - np.floor(244 / 2)
        xx, yy = np.meshgrid(x, y)
        self.mask244 = (np.sqrt((xx * xx) + (yy * yy)) - 244 / 2) > -3
        self.p_rot = p_rot

    def __call__(self, batch):
        images = []
        labels = torch.rand(size=(len(batch), 1)) > self.p_rot
        for i in zip(batch,labels):
            img = self.res(F_tv.pil_to_tensor(i[0]['image']))
            if i[1]:
                img = self.rr(img)
            if img.shape[0] == 1:
                img = torch.concat(3*[img],0)
            if img.shape[0] == 3:
                images.append(img)
        labels = torch.squeeze(labels.float())
        data = torch.stack(images,0)
        data[:, :, self.mask244] = 0
        data = data.float()
        return data, labels


def run():
    if torch.cuda.is_available():
        device = torch.cuda.device(0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    learning_rate = 0.0005
    batch_size = 24
    num_epochs = 10


    dataset_train = load_dataset("ILSVRC/imagenet-1k", cache_dir="./dataset", split="train")
    dataset_test = load_dataset("ILSVRC/imagenet-1k", cache_dir="./dataset", split="test")

    imgNet_collate = ImgNetCollator()
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, collate_fn=imgNet_collate)
    test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True, collate_fn=imgNet_collate)

    model = BooCNN(in_channels=3)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            targets = targets.to(device)

            scores = model(data)
            loss = criterion(scores, targets)
            if batch_index%100 == 0:
                print(f"current loss: {loss}")
                print(f"current scores: {scores}")

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

    def check_accuracy(loader, model):

        if loader.dataset.train:
            print("Checking accuracy on training data")
        else:
            print("Checking accuracy on test data")

        num_correct = 0
        num_samples = 0
        model.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)

                scores = model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)


            accuracy = float(num_correct) / float(num_samples) * 100
            print(f"Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")

        model.train()

    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)


if __name__ == '__main__':
    run()




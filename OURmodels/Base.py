import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import config as cfg

class Base(nn.Module):
    def __init__(self, device, decision_window):
        super(Base, self).__init__()
        self.model = base(decision_window)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def train(self, train_loader, device, epoch, N_epoch):
        self.model.train()
        count = 0
        all_correct = 0
        for iter, (X,y,z) in enumerate(tqdm.tqdm(train_loader, position=0, leave=True), start=1):
            convScore = self.model(X[6].to(device), X[7].to(device), X[8].to(device), X[9].to(device), X[10].to(device))
            y = y.to(device)
            count += convScore.shape[0]
            loss = nn.CrossEntropyLoss()(convScore, y)

            pred = torch.max(convScore, 1)[1]
            correct = (pred == y).sum().item()
            all_correct += correct

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print(f"ConcatNet Train Accuracy: {all_correct/count}")

    def test(self, test_loader, device):
        self.model.eval()
        with torch.no_grad():
            result = np.array([])
            gt = np.array([])
            for iter, (X,y,z) in enumerate(tqdm.tqdm(test_loader, position=0, leave=True), start=1):
                convScore = self.model(X[6].to(device), X[7].to(device), X[8].to(device), X[9].to(device), X[10].to(device))
                result = np.append(result, torch.max(convScore, 1)[1].cpu().numpy())
                gt = np.append(gt, y.cpu().numpy())
        return result, gt



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.LayerNorm([10,11], elementwise_affine=False)
        self.relu = nn.ReLU()

    def forward(self, X):
        y = self.norm(X)
        y = self.conv(y)
        y = self.relu(y)
        return y

class DenseBlock(nn.Module):

    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        in_c = in_channels
        self.norm = nn.LayerNorm([10,11], elementwise_affine=False)
        self.conv1 = ConvBlock(in_c, out_channels)
        in_c = in_c + out_channels
        self.conv2 = ConvBlock(in_c, out_channels)
        in_c = in_c + out_channels
        self.conv3 = ConvBlock(in_c, out_channels)
        in_c = in_c + out_channels
        self.conv4 = ConvBlock(in_c, out_channels)

        self.out_channels = in_channels + num_convs * out_channels # get the out channels

    def forward(self, X):
        y1 = self.conv1(X)
        X = torch.cat((X, y1), dim=1)
        y2 = self.conv2(X)
        X = torch.cat((X, y2), dim=1)
        y3 = self.conv3(X)
        X = torch.cat((X, y3), dim=1)
        y4 = self.conv4(X)
        y = torch.cat((X, y4), dim=1)

        return y


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels,pool_size = 7):
        super(TransitionBlock, self).__init__()
        self.norm = nn.LayerNorm([10,11], elementwise_affine=False)
        self.relu = nn.ReLU()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        # self.pool = nn.AvgPool3d(kernel_size=(pool_size, 1, 1), stride=(max(1,pool_size//2), 1, 1))

    def forward(self, X):
        y = self.norm(X)
        y = self.relu(y)
        y = self.conv(y)
        # y = self.pool(y)

        return y


class DenseNet_3D_t(nn.Module):
    def __init__(self,decision_window):
        super(DenseNet_3D_t, self).__init__()
        self.norm = nn.LayerNorm([10,11], elementwise_affine=False)
        # self.fc = nn.Linear(in_features=1*10*11, out_features=64)
        self.sigmoid = nn.Sigmoid()

        num_channels = 10

        # self.conv1 = nn.Conv3d(10, num_channels, kernel_size=(3, 3, 3), stride=1, padding=(0, 1, 1))

        num_convs = 4

        growth_rate = 5
        self.DB1 = DenseBlock(num_convs, num_channels, growth_rate)
        num_channels = self.DB1.out_channels
        self.TB1 = TransitionBlock(num_channels, num_channels // 2)
        num_channels = num_channels // 2 + 5

        self.DB2 = DenseBlock(num_convs, num_channels, growth_rate)
        num_channels = self.DB2.out_channels
        self.TB2 = TransitionBlock(num_channels, num_channels // 2)
        num_channels = num_channels // 2 + 5

        self.DB3 = DenseBlock(num_convs, num_channels, growth_rate)
        num_channels = self.DB3.out_channels
        self.TB3 = TransitionBlock(num_channels, num_channels // 2)
        num_channels = num_channels // 2 + 5

        self.DB4 = DenseBlock(num_convs, num_channels, growth_rate)
        num_channels = self.DB4.out_channels
        self.TB4 = TransitionBlock(num_channels, num_channels // 2)
        num_channels = num_channels // 2 + 5

        self.avgpool = nn.AvgPool3d(kernel_size=(decision_window, 1, 1), stride=1)
        self.cnn = nn.Conv3d(num_channels, 1, kernel_size=1)

    def forward(self, x):
        # Layer 1
        x = self.norm(x)
        eeg_raw = x
        # x = x.unsqueeze(dim=1)

        x = torch.cat((x, eeg_raw), dim=1)
        x = self.DB1(x)
        x = self.TB1(x)

        x = torch.cat((x, eeg_raw), dim=1)
        x = self.DB2(x)
        x = self.TB2(x)

        x = torch.cat((x, eeg_raw), dim=1)
        x = self.DB3(x)
        x = self.TB3(x)

        x = torch.cat((x, eeg_raw), dim=1)
        x = self.DB4(x)
        x = self.TB4(x)

        # x = self.sigmoid(x)
        x = torch.cat((x, eeg_raw), dim=1)
        x = self.cnn(x)
        x = self.sigmoid(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        # x = self.fc(x)


        return x




class GNet(nn.Module):
    def __init__(self, N, dim):
        super(GNet, self).__init__()
        self.linear = nn.Linear(dim, N)

    def forward(self, x):
        x = self.linear(x)
        return x

class FeatureCNN(nn.Module):
    def __init__(self, decision_window):
        super(FeatureCNN, self).__init__()
        self.tcnn = DenseNet_3D_t(decision_window)
        # self.fc = nn.Linear(128, 64)

    def forward(self, x):
        # x1 = x1.unsqueeze(dim=1)
        # x2 = x2.unsqueeze(dim=1)
        # x3 = x3.unsqueeze(dim=1)
        # x4 = x4.unsqueeze(dim=1)
        # x5 = x5.unsqueeze(dim=1)
        # x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.tcnn(x)

        # x = self.fc(x1)

        return x

# class FeatureCNN(nn.Module):
#     def __init__(self, decision_window):
#         super(FeatureCNN, self).__init__()
#         self.tcnn = DenseNet_3D_t()
#         self.fcnn = DenseNet_3D_f()
#         self.fc = nn.Linear(128, 64)
#
#
#     def forward(self, x1, x2):
#         x1 = self.tcnn(x1)
#         x2 = self.fcnn(x2)
#         x = torch.cat((x1, x2), dim=1)
#         x = self.fc(x)
#         return x


class base(nn.Module):
    def __init__(self, decision_window):
        super(base, self).__init__()
        self.feature_cnn = FeatureCNN(decision_window)
        self.g_net = GNet(2, 110)

    def forward(self, x1, x2, x3, x4, x5):
        x1 = x1.unsqueeze(dim=1)
        x2 = x2.unsqueeze(dim=1)
        x3 = x3.unsqueeze(dim=1)
        x4 = x4.unsqueeze(dim=1)
        x5 = x5.unsqueeze(dim=1)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.feature_cnn(x)
        out = self.g_net(x)
        return out



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    decision_window = 64
    model = base(decision_window).to(device)

    x1 = torch.rand((10, decision_window,10,11)).to(device)
    x2 = torch.rand((10, decision_window,10,11)).to(device)
    x3 = torch.rand((10, decision_window,10,11)).to(device)
    x4 = torch.rand((10, decision_window,10,11)).to(device)
    x5 = torch.rand((10, decision_window,10,11)).to(device)

    out = model(x1, x2, x3, x4, x5)
    summary(model, [(decision_window,10,11), (decision_window,10,11), (decision_window,10,11), (decision_window,10,11), (decision_window,10,11)])

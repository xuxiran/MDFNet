import numpy as np
import torch
import torch.nn as nn
import tqdm
import config as cfg

# no 5 band
class Notem(nn.Module):
    def __init__(self, device, decision_window):
        super(Notem, self).__init__()
        self.model = base(1)
        self.num_classes = 2
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    def train(self, train_loader, device, epoch, N_epoch):
        self.model.train()
        count = 0
        all_correct = 0
        for iter, (X,label,identities) in enumerate(tqdm.tqdm(train_loader, position=0, leave=True), start=1):

            all_y = label.to(device)
            output = self.model(X[3][:,5:10,:,:].to(device))
            all_p = output[1] if isinstance(output, tuple) else output

            loss = torch.nn.functional.cross_entropy(all_p, all_y)
            count += all_p.shape[0]
            pred = torch.max(all_p, 1)[1]
            correct = (pred == all_y).sum().item()
            all_correct += correct


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print(f"Notem Train Accuracy: {all_correct/count}")


    def test(self, test_loader, device):
        self.model.eval()
        with torch.no_grad():
            result = np.array([])
            gt = np.array([])
            for iter, (X,y,z) in enumerate(tqdm.tqdm(test_loader, position=0, leave=True), start=1):
                output = self.model(X[3][:,5:10,:,:].to(device))
                convScore = output[1] if isinstance(output, tuple) else output
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
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1))
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
        x = x.unsqueeze(dim=2)
        x = self.norm(x)
        eeg_raw = x


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
        # x = self.avgpool(x)
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

        x = self.tcnn(x)


        return x



class base(nn.Module):
    def __init__(self, decision_window):
        super(base, self).__init__()
        self.feature_cnn = FeatureCNN(decision_window)
        self.g_net = GNet(2, 110)

    def forward(self, x):

        x = self.feature_cnn(x)
        out = self.g_net(x)
        return x,out



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    decision_window = 64
    model = base(5).to(device)

    x = torch.rand((10, 5,10,11)).to(device)
    out = model(x)
    summary(model,(5,10,11))

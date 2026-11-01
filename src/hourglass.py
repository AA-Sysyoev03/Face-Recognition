import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

        self.conv1 = nn.Conv2d(
            in_channels, out_channels // 2, kernel_size=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels // 2)

        self.conv2 = nn.Conv2d(
            out_channels // 2,
            out_channels // 2,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels // 2)

        self.conv3 = nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        return self.relu(x + residual)


class Hourglass(nn.Module):
    def __init__(self, depth, f, increase=0):
        super().__init__()

        self.depth = depth
        nf = f + increase

        self.up1 = ResidualBlock(f, f)

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.low1 = ResidualBlock(f, nf)

        if depth > 1:
            self.low2 = Hourglass(
                depth - 1, nf, increase
            )
        else:
            self.low2 = ResidualBlock(nf, nf)

        self.low3 = ResidualBlock(nf, f)

        self.up2 = nn.Upsample(
            scale_factor=2,
            mode="nearest"
        )

    def forward(self, x):
        up1 = self.up1(x)

        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)

        up2 = self.up2(low3)

        if up1.shape[-2:] != up2.shape[-2:]:
            up2 = F.interpolate(
                up2,
                size=up1.shape[-2:],
                mode="nearest"
            )

        return up1 + up2


class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()

        self.conv = nn.Conv2d(
            x_dim,
            y_dim,
            kernel_size=1,
            bias=False
        )

        self.bn = nn.BatchNorm2d(y_dim)

    def forward(self, x):
        return self.bn(self.conv(x))


class HeatmapLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        return ((pred - gt) ** 2).mean()


class StackedHourglass(nn.Module):
    def __init__(
        self,
        nstack=2,
        inp_dim=128,
        oup_dim=5,
        increase=0
    ):
        super().__init__()

        self.nstack = nstack

        # Начальная обработка RGB-изображения.
        # Размер 256x256 преобразуется в 64x64.
        self.pre = nn.Sequential(
            nn.Conv2d(
                3, 64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            ResidualBlock(64, 128),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            ResidualBlock(128, 128),
            ResidualBlock(128, inp_dim)
        )

        # Несколько последовательных Hourglass.
        self.hgs = nn.ModuleList([
            Hourglass(
                depth=3,
                f=inp_dim,
                increase=increase
            )
            for _ in range(nstack)
        ])

        # Обработка признаков каждого Hourglass.
        self.features = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(inp_dim, inp_dim),

                nn.Conv2d(
                    inp_dim,
                    inp_dim,
                    kernel_size=1,
                    bias=False
                ),

                nn.BatchNorm2d(inp_dim),
                nn.ReLU(inplace=True)
            )
            for _ in range(nstack)
        ])

        # Предсказание пяти карт ключевых точек.
        self.outs = nn.ModuleList([
            nn.Conv2d(
                inp_dim,
                oup_dim,
                kernel_size=1
            )
            for _ in range(nstack)
        ])

        # Связи между последовательными Hourglass.
        self.merge_features = nn.ModuleList([
            Merge(inp_dim, inp_dim)
            for _ in range(nstack - 1)
        ])

        self.merge_preds = nn.ModuleList([
            Merge(oup_dim, inp_dim)
            for _ in range(nstack - 1)
        ])

        self.heatmap_loss = HeatmapLoss()

    def forward(self, x):
        x = self.pre(x)

        combined_hm_preds = []

        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)

            combined_hm_preds.append(preds)

            if i < self.nstack - 1:
                x = (
                    x
                    + self.merge_preds[i](preds)
                    + self.merge_features[i](feature)
                )

        return torch.stack(
            combined_hm_preds,
            dim=1
        )

    def calc_loss(self, outputs, targets):
        total_loss = 0.0
        nstacks = outputs.size(1)

        for i in range(nstacks):
            pred = outputs[:, i]
            total_loss += self.heatmap_loss(
                pred, targets
            )

        return total_loss / nstacks
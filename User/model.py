import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()

        self.down_block_1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=48, kernel_size=3, stride=1, padding='same', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv1d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding='same', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.AvgPool1d(2, 2)
        )

        self.down_block_2 = nn.Sequential(
            nn.Conv1d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding='same', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.AvgPool1d(2, 2)
        )

        self.down_block_3 = nn.Sequential(
            nn.Conv1d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding='same', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.AvgPool1d(2, 2)
        )

        self.down_block_4 = nn.Sequential(
            nn.Conv1d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding='same', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.AvgPool1d(2, 2)
        )

        self.down_block_5 = nn.Sequential(
            nn.Conv1d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding='same', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.AvgPool1d(2, 2),
            nn.Conv1d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding='same', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        self.up_block_1 = nn.Sequential(
            nn.Conv1d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding='same', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv1d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding='same', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        self.up_block_2 = nn.Sequential(
            nn.Conv1d(in_channels=144, out_channels=96, kernel_size=3, stride=1, padding='same', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv1d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding='same', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        self.up_block_3 = nn.Sequential(
            nn.Conv1d(in_channels=144, out_channels=96, kernel_size=3, stride=1, padding='same', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv1d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding='same', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        self.up_block_4 = nn.Sequential(
            nn.Conv1d(in_channels=144, out_channels=96, kernel_size=3, stride=1, padding='same', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv1d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding='same', bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        self.up_block_5 = nn.Sequential(
            nn.Conv1d(in_channels=96 + in_channel, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),

            nn.Conv1d(in_channels=32, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=True)
        )

    def forward(self, x):
        down_1 = self.down_block_1(x)
        down_2 = self.down_block_2(down_1)
        down_3 = self.down_block_3(down_2)
        down_4 = self.down_block_4(down_3)
        up_1 = self.down_block_5(down_4)

        concat_1 = torch.cat((up_1, down_4), dim=1)
        up_2 = self.up_block_1(concat_1)
        concat_2 = torch.cat((up_2, down_3), dim=1)
        up_3 = self.up_block_2(concat_2)
        concat_3 = torch.cat((up_3, down_2), dim=1)
        up_4 = self.up_block_3(concat_3)
        concat_4 = torch.cat((up_4, down_1), dim=1)
        up_5 = self.up_block_4(concat_4)
        concat_5 = torch.cat((up_5, x), dim=1)
        out = self.up_block_5(concat_5)

        return out


if __name__ == '__main__':
    x = torch.zeros(10, 1, 1024)
    model = UNet(1, 1)
    y = model(x)
    print(y.shape)

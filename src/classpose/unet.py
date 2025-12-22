import torch


class UNetBlock(torch.nn.Module):
    """
    A basic UNet block.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize the UNetBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(UNetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.relu = torch.nn.ReLU()

    def forward(
        self, x: torch.Tensor, skip_last_activation: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of the UNetBlock.

        Args:
            x (torch.Tensor): Input tensor.
            skip_last_activation (bool): Whether to skip the last activation.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if not skip_last_activation:
            x = self.relu(x)
        return x


class UNetBlockDown(torch.nn.Module):
    """
    A UNet block that downsample the input.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize the UNetBlockDown.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(UNetBlockDown, self).__init__()
        self.block = UNetBlock(in_channels, out_channels)
        self.downconv = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=2, stride=2
        )

    def forward(
        self, x: torch.Tensor, skip_last_activation: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of the UNetBlockDown.

        Args:
            x (torch.Tensor): Input tensor.
            skip_last_activation (bool): Whether to skip the last activation.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.block(x, skip_last_activation=skip_last_activation)
        x_down = self.downconv(x)
        return x, x_down


class UNetBlockUp(torch.nn.Module):
    """
    A UNet block that upsample the input.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize the UNetBlockUp.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(UNetBlockUp, self).__init__()
        self.block = UNetBlock(in_channels, out_channels)
        self.upconv = torch.nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=2, stride=2
        )

    def forward(
        self, x: torch.Tensor, skip_last_activation: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of the UNetBlockUp.

        Args:
            x (torch.Tensor): Input tensor.
            skip_last_activation (bool): Whether to skip the last activation.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.block(x, skip_last_activation=skip_last_activation)
        x = self.upconv(x)
        return x


class UNet(torch.nn.Module):
    """
    A UNet network.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_channels: list[int] = [64, 128, 256, 512],
    ):
        """
        Initialize the UNet.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            n_channels (list[int]): Number of channels in each block.
        """
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_channels = n_channels
        self.initialise_network()

    def initialise_network(self):
        in_channel_sequence = [self.in_channels, *self.n_channels]
        out_channel_sequence = [*self.n_channels[::-1], self.out_channels]
        self.encoder_blocks = torch.nn.ModuleList(
            [
                UNetBlockDown(in_channels, out_channels)
                for in_channels, out_channels in zip(
                    in_channel_sequence[:-1], in_channel_sequence[1:]
                )
            ]
        )
        self.decoder_blocks = torch.nn.ModuleList(
            [
                UNetBlockUp(in_channels * 2, out_channels)
                for in_channels, out_channels in zip(
                    out_channel_sequence[:-1], out_channel_sequence[1:]
                )
            ]
        )

        self.bottleneck_down = UNetBlockDown(
            in_channel_sequence[-1], in_channel_sequence[-1]
        )
        self.bottleneck_up = UNetBlockUp(
            in_channel_sequence[-1], in_channel_sequence[-1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UNet.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        encoded_features = []
        for block in self.encoder_blocks:
            _, x = block(x)
            encoded_features.append(x)
        encoded_features = encoded_features[::-1]
        _, x = self.bottleneck_down(x)
        x = self.bottleneck_up(x)
        n_dec = len(self.decoder_blocks)
        for i, block in enumerate(self.decoder_blocks):
            x = block(
                torch.cat((x, encoded_features[i]), dim=1),
                skip_last_activation=i == (n_dec - 1),
            )
        return x


if __name__ == "__main__":
    in_channels = 128
    out_channels = 32
    unet = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        n_channels=[64, 130, 256],
    )
    x = torch.randn(1, in_channels, 256, 256)
    y = unet(x)
    print(y.shape, y.min(), y.max(), y.shape[1] == out_channels)

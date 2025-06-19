import torch
import torch.nn as nn

class SnakeV0(nn.Module):
    def __init__(self, in_channels: int, hidden_units: int, output_shape: int):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Calcular tamaño de salida para la capa lineal con un tensor dummy
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 480, 640)
            out = self.conv_block_1(dummy_input)
            out = self.conv_block_2(out)
            flattened_size = out.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
    
    

class SnakeV1(nn.Module):
    def __init__(self, in_channels: int, hidden_units: int, output_shape: int):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Imagen original: [3, 640, 480] -> tras 2 MaxPool2d con kernel_size=2: tamaño = [hidden_units, 160, 120]
        conv_output_size = hidden_units * 30 * 22

        self.classifier = nn.Sequential(
            nn.Linear(conv_output_size + 2, 128),  # Añadimos 2 (x, y)
            nn.ReLU(),
            nn.Linear(128, output_shape)
        )

    def forward(self, image, food_pos):
        x = self.conv_block_1(image)
        x = self.conv_block_2(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat([x, food_pos], dim=1)  # Concatenamos posición comida
        x = self.classifier(x)
        return x
    
    
class SnakeV2(nn.Module):
    def __init__(self, in_channels: int, hidden_units: int, output_shape: int):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Imagen original: [3, 640, 480] -> tras 2 MaxPool2d con kernel_size=2: tamaño = [hidden_units, 160, 120]
        conv_output_size = hidden_units * 30 * 22

        self.classifier = nn.Sequential(
            nn.Linear(conv_output_size + 8, 128),
            nn.ReLU(),
            nn.Linear(128, output_shape)
        )

    def forward(self, image, info_tensor):
        x = self.conv_block_1(image)
        x = self.conv_block_2(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat([x, info_tensor], dim=1)
        x = self.classifier(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel




class DyffusionTinyConfig(BaseModel):
    horizon: int = 5


class DyffusionTinyForecaster(nn.Module):
    def __init__(self, config: DyffusionTinyConfig):
        super().__init__()
        self.config = config
        # Time Branch
        self.time_embedding = nn.Embedding(config.horizon, 64)
        self.dense = nn.Linear(64, 64)
        self.dense_1 = nn.Linear(64, 64)
        self.film_gamma = nn.Linear(64, 32)
        self.film_beta = nn.Linear(64, 32)
        
        # Image Branch (Assuming Channels First for PyTorch: 1, 120, 321)
        self.conv2d = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.2)
        
        # Final Layers
        self.conv2d_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn_1 = nn.BatchNorm2d(32)
        self.conv_stack = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, image_input, i_index):
        # Time processing
        t = self.time_embedding(i_index)
        t = self.dense(t)
        t = self.dense_1(t)
        
        gamma = self.film_gamma(t).view(-1, 32, 1, 1)
        beta = self.film_beta(t).view(-1, 32, 1, 1)
        
        # Image processing
        img = self.conv2d(image_input)
        img = self.bn(img)
        img = self.dropout(img)
        
        # FiLM Fusion
        x = (gamma * img) + beta
        x = F.relu(x)
        
        # Rest of the network
        x = self.conv2d_1(x)
        x = self.bn_1(x)
        x = self.conv_stack(x)
        return self.final_conv(x)


class DiffysionTinyInterpolator(nn.Module):
    def __init__(self, config: DyffusionTinyConfig):
        super().__init__()
        self.config = config
        
        # Time/Conditioning Branch
        self.time_embedding = nn.Embedding(config.horizon, 64)
        self.dense = nn.Linear(64, 64)
        self.dense_1 = nn.Linear(64, 64)
        self.film_gamma = nn.Linear(64, 32)
        self.film_beta = nn.Linear(64, 32)
        
        # Image Branch (2 input channels)
        # Note: PyTorch uses (Batch, Channels, Height, Width)
        self.conv2d = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.2)
        
        # Final Output Layer
        self.conv2d_1 = nn.Conv2d(32, 1, kernel_size=3, padding=1)


    def forward(self, image_input, i_index):
        # 1. Process conditioning
        t = self.time_embedding(i_index)
        t = self.dense(t)
        t = self.dense_1(t)
        
        # Reshape gamma/beta for broadcasting: (Batch, 32, 1, 1)
        gamma = self.film_gamma(t).unsqueeze(-1).unsqueeze(-1)
        beta = self.film_beta(t).unsqueeze(-1).unsqueeze(-1)
        
        # 2. Process Image features
        img = self.conv2d(image_input)
        img = self.bn(img)
        img = self.dropout(img)
        
        # 3. Apply FiLM
        x = (gamma * img) + beta
        x = F.relu(x)
        
        # 4. Final Conv
        return self.conv2d_1(x)

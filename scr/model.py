import torch.nn as nn
import torch.nn.functional as F

class KyrgyzLetterCNN(nn.Module):
    def __init__(self):
        super(KyrgyzLetterCNN, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 50 → 25

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 25 → 12

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 12 → 6

            nn.Flatten(),  # [128, 6, 6] → 4608

            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 36)  # ✅ 36 классов для киргизского алфавита
        )

    def forward(self, x):
        return self.model(x)


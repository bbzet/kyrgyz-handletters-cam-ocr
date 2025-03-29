import torch
from torch.utils.data import DataLoader
from scr.dataset import CustomKyrgyzDataset
from scr.model import KyrgyzLetterCNN
import pandas as pd
import torchvision.transforms as tfs
import os
from tqdm import tqdm

test_ids = pd.read_csv('data/test.csv')['id'].to_list()

test_transform = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize(mean=[0.5], std=[0.5]),
])

test_dataset = CustomKyrgyzDataset('data/test.csv', train=False, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = KyrgyzLetterCNN()
model.load_state_dict(torch.load('models/kyrgyzletters_model.pt', map_location=torch.device('cpu')))

model.eval()

kyrgyz_letters = [
    'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З',
    'И', 'Й', 'К', 'Л', 'М', 'Н', 'Ң', 'О', 'Ө',
    'П', 'Р', 'С', 'Т', 'У', 'Ү', 'Ф', 'Х', 'Ц',
    'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я'
]

id_to_char = {i: kyrgyz_letters[i] for i in range(36)}

predictions = []
with torch.no_grad():
    for images in tqdm(test_loader, desc='Предказания'):
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.numpy())

letters = [id_to_char[i] for i in predictions]

os.makedirs('predictions', exist_ok=True)
submission = pd.DataFrame({"Predictions": letters})
submission.to_csv('predictions/submission.csv', index=False)

submission_kaggle = pd.DataFrame({
    'id': test_ids,
    'label': [i + 1 for i in predictions]
})

submission_kaggle.to_csv('predictions/submission_kaggle.csv', index=False)

print("Предказания сохранены в файле predictions/submission.csv")

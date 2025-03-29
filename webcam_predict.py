import cv2
import torch
import torchvision.transforms as tfs
from scr.model import KyrgyzLetterCNN
import numpy as np

kyrgyz_letters = [
    'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З',
    'И', 'Й', 'К', 'Л', 'М', 'Н', 'Ң', 'О', 'Ө',
    'П', 'Р', 'С', 'Т', 'У', 'Ү', 'Ф', 'Х', 'Ц',
    'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я'
]

id_to_char = {i: kyrgyz_letters[i] for i in range(36)}

transform = tfs.Compose([
    tfs.ToPILImage(),
    tfs.Resize((50, 50)),
    tfs.ToTensor(),
    tfs.Normalize(mean=[0.5], std=[0.5])
])


model = KyrgyzLetterCNN()
model.load_state_dict(torch.load('models/kyrgyzletters_model.pt', map_location=torch.device('cpu')))
model.eval()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Не удалось открыть камеру")
    exit()

print("\n Камера запущена")
print("Правила:")
print("Помести букву в рамку")
print("Перед нажатием убедитесь, что клавиатура в ENG английской раскладке")
print("Нажмите 'p' - распознать,\n        'q' - выйти")


while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр")
        break

    h, w = frame.shape[:2]
    size = 200
    x1 = w // 2 - size // 2
    y1 = h // 2 - size // 2
    x2 = x1 + size
    y2 = y1 + size

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):
        roi = frame[y1:y2, x1:x2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15,
            C=10
        )

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # удаляет шум
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # заполняет пробелы

        cv2.imshow("Processed", thresh)

        img_tensor = transform(thresh).unsqueeze(0)  # [1, 1, 50, 50]


        print(img_tensor.shape)
        print(img_tensor.min(), img_tensor.max())
        with torch.no_grad():
            output = model(img_tensor)
            _, pred = torch.max(output, 1)
            char = id_to_char[pred.item()]

        cv2.putText(frame, f"Label: {pred.item() + 1}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Kyrgyz Letter Recognition", frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
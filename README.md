# 🖋️ Kyrgyz Handwritten Letter Recognition

Распознавание рукописных **кыргызских букв** в реальном времени через веб-камеру с использованием PyTorch и OpenCV.


## 📂 Структура проекта

```bash
handwritten-kyrgyz-letters/
├── notebooks/
│   └── 1_data_visualization.ipynb  # Анализ и визуализация данных
├── src/
│   ├── dataset.py                  # PyTorch Dataset
│   ├── model.py                    # CNN модель
│   ├── train.py                    # Обучение модели
├── predictions/
│   └── submission_kaggle.csv  # Результаты предсказаний для кагла +
│   └── submission.csv              # Результаты предсказаний
├── webcam_predict.py           # Предсказание с веб-камеры
├── main.py                              # Запуск камеры
├── predictions.py                    # Для предсказании
├── requirements.txt                # Зависимости проекта
├── .gitignore
└── README.md

```
## 📥 Данные

🔗 Скачать датасет можно [по ссылке](https://www.kaggle.com/competitions/kyrgyz-language-hand-written-letter-kyrgyz-mnist/data). CSV содержит 2501 столбец: `label`, `pixel_0` … `pixel_2499`.


---

## 🚀 Запуск модели

```bash
# Установка зависимостей
pip install -r requirements.txt

# Обучение модели (на CPU)
python src/train.py

# Предсказание на тестовой выборке (сохранит CSV)
python predictions.py

# Запуск в реальном времени с веб-камеры
python webcam_predict.py
```

## Input
```bash
📷 Handwritten Kyrgyz Letter Recognition
1 — Запустить веб-камеру
Другое — Выйти
Введите номер: 1

 Камера запущена
Правила:
Помести букву в рамку
Перед нажатием убедитесь, что клавиатура в ENG английской раскладке
Нажмите 'p' - распознать,
        'q' - выйти
```
---

##  Формат выходных данных

В режиме веб-камеры модель показывает **label** предсказанной буквы:

```
Label: 5  →  (означает 5-я буква в алфавите: 'Д')
```

---


## 📊 Метки и алфавит

В проекте используется **36 кыргызских букв**. Метка `label` — это число от 1 до 36:

```
1 → А, 2 → Б, 3 → В, ..., 36 → Я
```

---
**Точность модели**
Обученная модель сверточной нейронной сети (CNN) показала следующие результаты:

- Точность на валидации: 89.28%

- Точность на тестовой выборке: 95.00%
![](https://raw.githubusercontent.com/bbzet/kyrgyz-handletters-cam-ocr/refs/heads/main/screenshots/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202025-03-29%20181014.png)

Модель уверенно обобщает данные и подходит для реального применения — распознавания рукописных кыргызских букв в режиме реального времени через веб-камеру.


**Зависимости**
> Все зависимости описаны в `requirements.txt`


# ML Application with Streamlit

Цей проект є веб-додатком для роботи з машинним навчанням, що дозволяє тренувати та тестувати моделі CNN і VGG16 на основі набору даних Fashion MNIST. Додаток створений з використанням **Streamlit**.

## 📋 Опис

Додаток має наступні функції:
1. Тренування моделі CNN на чорно-білих зображеннях (Fashion MNIST).
2. Тренування моделі VGG16 на кольорових зображеннях (перетворених з Fashion MNIST).
3. Візуалізація результатів навчання, включаючи графіки втрат і точності.
4. Тестування зображень, завантажених користувачем, на основі обраної моделі.

## 🛠️ Структура файлів

- **`main.py`**: Скрипт для запуску Streamlit-додатку.
- **`app.py`**: Основний модуль, що містить функціонал додатку, включаючи визначення моделей, тренування, візуалізацію та тестування.

## Запуск додатку

```bash
python main.py
```

Або напряму:

```bash
streamlit run main.py
```

## 📊 Моделі

# CNN:

Використовує 2D конволюційні шари.
Архітектура включає Dropout для запобігання перенавчанню.
Тренується на зображеннях розміром 28x28 у градаціях сірого.

# VGG16:

Попередньо тренована на ImageNet.
Вхідні зображення масштабуються до розміру 32x32 з трьома каналами (RGB).
Використовується лише головний блок VGG16 (заморожений).

## 🖼️ Як працювати з додатком

1. Навчання моделей
Оберіть CNN або VGG16 у меню.
Введіть кількість епох та натисніть "Тренувати модель".
2. Тестування
Завантажте зображення у форматі .jpg, .jpeg, .png або .gif.
Оберіть модель (CNN або VGG16).
Натисніть "Розпізнати зображення".
3. Збереження моделей
Треновані моделі зберігаються у папці saveModels у форматі .keras.

## 📂 Приклад результату

# Графіки навчання

Графік втрат (loss) та точності (accuracy).

# Розпізнавання зображень

Зображення з передбаченням класу та гістограма ймовірностей.

## 👨‍💻 Автор

Автор проекту: Сергій Труш
import streamlit as st
from streamlit_option_menu import option_menu # pip install streamlit-option-menu
from PIL import Image, ImageOps
import os
import numpy as np
import keras
from keras import layers
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import random

epochsSetCNN = 5
maxEpochSetCNN = 100
batchSetCNN = 512
learningRateSetCNN = 0.00045
decayRateSetCNN = 0.0015

epochsSetVGG16 = 5
maxEpochSetVGG16 = 100
batchSetVGG16 = 512
learningRateSetVGG16 = 0.01
decayRateSetVGG16 = 0.001

showDatasetSamples = 10 # Кількість семплів для перевірки після навчання

traineCNN = False
traineVGG16 = False

progressCalcCNN = 0 # Логічний прапорець для визначення, чи тренувати CNN модель
progressCalcVGG16 = 0 # Логічний прапорець для визначення, чи тренувати VGG16 модель

# Поточна робоча директорія
current_dir = os.getcwd()

# Список імен класів датасету
classes = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

# Функція для завантаження датасету. Приймає аргумент isCNN, який визначає, чи це CNN модель.
def datasetDownload(isCNN):
    # Завантажуємо набір даних fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
    print("Dataset is download now.")

    if(isCNN == True):
        # Розширюємо розмірність зображень для додання каналу (для конволюційних шарів)
        train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
        test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    else:
        # Змінюємо тип даних на float32 для збільшення швидкості обчислень
        train_images = train_images.astype('float32')
        test_images = test_images.astype('float32')

        # Змінюємо розмір зображень на (32, 32, 3) для відповідності VGG16
        train_images = np.repeat(train_images[..., np.newaxis], 3, -1)  # RGB-канал
        test_images = np.repeat(test_images[..., np.newaxis], 3, -1)
        train_images = tf.image.resize(train_images, (32, 32))
        test_images = tf.image.resize(test_images, (32, 32))

        # Перетворюємо мітки у вектор одного з n (10 для fashion_mnist)
        train_labels = to_categorical(train_labels, 10)
        test_labels = to_categorical(test_labels, 10)
    
    return train_images, train_labels, test_images, test_labels

# Функція для створення моделі CNN
def createCNNModel():
    # Створюємо покращену нейронну мережу з конволюційними шарами
    modelCNN = models.Sequential([
        # Перший конволюційний шар
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Перетворюємо вектор ознак у плоский шар перед застосуванням до повністю з'єднаних шарів
        layers.Flatten(),
        
        # Додаємо повний з'єднаний шар з Dropout для запобігання перенавчанню
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='linear'),
        layers.Dropout(0.6),
        layers.Dense(10, activation='softmax')
    ])

    # Компілюємо модель
    modelCNN.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    
    return modelCNN

# Функція для створення моделі VGG16
def createVGG16Model():
    conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
    conv_base.trainable = False

    # Створюємо покращену нейронну мережу з конволюційними шарами
    modelVGG16 = models.Sequential([
        conv_base,
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        # Вихідний шар з 10 класами
        layers.Dense(10, activation='softmax')
    ])

    # Компілюємо модель
    modelVGG16.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=learningRateSetVGG16),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return modelVGG16

# Експоненційне зменшення швидкості навчання для CNN моделі
def exponential_decayCNN(epoch, lr):
    global progressCalcCNN
    progressCalcCNN = int(((epoch + 1) / epochsSetCNN) * 100)
    progressCNNBar.progress(progressCalcCNN, text= f"Прогрес навчання {progressCalcCNN}%")  # Оновити progress bar

    if(epoch > 0):  
        decay_rate = decayRateSetCNN
        return float(lr * tf.math.exp(-decay_rate * epoch))
    else:
        return learningRateSetCNN   

# Експоненційне зменшення швидкості навчання для VGG16 моделі
def exponential_decayVGG16(epoch, lr):
    global progressCalcVGG16
    progressCalcVGG16 = int(((epoch + 1) / epochsSetVGG16) * 100)
    progressVGG16Bar.progress(progressCalcVGG16, text= f"Прогрес навчання {progressCalcVGG16}%")  # Оновити progress bar

    if(epoch > 0):    
        decay_rate = decayRateSetVGG16
        return float(lr * tf.math.exp(-decay_rate * epoch))
    else:
        return learningRateSetVGG16   

# Функція для візуалізації графіку точності та втрат
def showPlot(traineHistory):
    # Візуалізуємо графік втрат і точності
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(traineHistory.history['loss'], label='Втрати на тренувальних даних')
    plt.plot(traineHistory.history['val_loss'], label='Втрати на валідаційних даних')
    plt.xlabel('Епохи')
    plt.ylabel('Точність')
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(traineHistory.history['accuracy'], label='Точність на тренувальних даних')
    plt.plot(traineHistory.history['val_accuracy'], label='Точність на валідаційних даних')
    plt.xlabel('Епохи')
    plt.ylabel('Точність')
    plt.legend(loc='lower right')
    return plt

def lablesPlot(predictedLablees):
    # Переконайтеся, що це одномірний масив
    predictedLablees_ = np.squeeze(np.array(predictedLablees))

    # Індекс класу з найвищою ймовірністю
    predictedClasses = np.argmax(predictedLablees_)

    # Динамічне масштабування осі y
    y_min = max(0, min(predictedLablees_))  # Відступ вниз (але не нижче 0)
    y_max = min(1.1, max(predictedLablees_))  # Відступ вгору (але не вище 1.1)

    # Створення графіка
    plt.figure(figsize=(10, 5))
    bars = plt.bar(classes, predictedLablees, color='gray')

    # Виділення передбаченого класу кольором
    bars[predictedClasses].set_color('orange')

    # Додаткові стилі
    plt.title(f'Прогнозований клас: {classes[predictedClasses]}', fontsize=16, color='blue')
    plt.xlabel('Класи', fontsize=12)
    plt.ylabel('Ймовірність', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(y_min, y_max)  # Застосування автоматичного масштабування
    plt.tight_layout()
    return plt

# Очищаємо графік для наступного використання
def plotsClear():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


# Встановлюємо заголовок для веб-додатку
st.title('HW 16')

# Створюємо навігаційне меню з допомогою streamlit_option_menu
selected = option_menu(menu_title=None, options=["CNN", "VGG16", "Models test"],
                       default_index=2,
                       orientation="horizontal")

# Якщо вибрано опцію "CNN"
if selected == "CNN":
    st.title("Навчання CNN")
    
    # Створюємо інпут для введення кількості епох навчання (обмежений діапазоном)
    epochsSetCNN = st.number_input(label = f'Введіть кількість епох навчання CNN(2..{maxEpochSetCNN})', min_value = 2, max_value = maxEpochSetCNN, value = epochsSetCNN, step = 1)
    
    # Створюємо кнопку для початку тренування моделі CNN
    traineCNN = st.button('Тренувати модель CNN', disabled = traineVGG16)

    print("epochsSetCNN =", epochsSetCNN)
    print("traineCNN =", traineCNN)

    # Якщо кнопка тренування натиснута
    if traineCNN:
        st.title('Триває навчання моделі...')
        
        # Створюємо progress bar для відображення прогресу тренування
        progressCNNBar = st.progress(progressCalcCNN, text= f"Прогрес навчання {progressCalcCNN}%")

        # Завантажуємо та підготовлюємо дані для CNN моделі
        train_imagesCNN, train_labels, test_imagesCNN, test_labels = datasetDownload(1)  # 1 - означає CNN
        
        # Створюємо модель CNN
        modelCNN = createCNNModel()
        
        # Додаємо callback для експоненційного зменшення швидкості навчання
        lr_schedulerCNN = tf.keras.callbacks.LearningRateScheduler(exponential_decayCNN)
        
        # Навчаємо модель
        historyCNN = modelCNN.fit(train_imagesCNN, train_labels, batch_size=batchSetCNN, epochs=epochsSetCNN, validation_data=(test_imagesCNN, test_labels), callbacks=[lr_schedulerCNN], verbose=0)
        
        st.success('Навчання завершено :) ')
        
        # Візуалізуємо графік точності та втрат
        st.pyplot(showPlot(historyCNN))
        
        # Очищаємо графік для наступного використання
        plotsClear()

        # Оцінюємо модель
        lossCNN, accuracyCNN = modelCNN.evaluate(test_imagesCNN, test_labels, verbose=0)
        st.write(f"Точність розпізнавання: {accuracyCNN:.2f}")

        # Зберігаємо модель у файл
        pathCNNModel = os.path.join(current_dir, "saveModels\\", "CNNModel.keras")
        modelCNN.save(pathCNNModel)

        st.title(f"Модель збережена за шляхом:")
        st.write(pathCNNModel)

        st.title('Тестування на датасеті')

        # Вибираємо випадкові зразки для тестування
        randomChoice = random.randint(0, len(test_labels) - showDatasetSamples)
        test_images = test_imagesCNN[randomChoice:randomChoice + showDatasetSamples]
        test_labels = test_labels[randomChoice:randomChoice + showDatasetSamples]
        
        # Прогнозуємо результати для зразків
        predicted_results = modelCNN.predict(test_images)
        predicted_labels = tf.argmax(predicted_results, 1).numpy()

        for sampleCNN in range(showDatasetSamples):
            plt.imshow(test_images[sampleCNN].reshape(28, 28), cmap='gray')
            st.write(f"Справжня мітка: {classes[test_labels[sampleCNN]]}, Прогнозована мітка: {classes[predicted_labels[sampleCNN]]}")
            st.pyplot(plt)
            # Очищаємо графік для наступного використання
            plotsClear()
            st.write("Графік прогнозів для міток")
            st.pyplot(lablesPlot(predicted_results[sampleCNN]))
            print("predicted_results[sampleCNN]", predicted_results[sampleCNN])
            # Очищаємо графік для наступного використання
            plotsClear()

        traineCNN = False
        
# Якщо вибрано опцію "VGG16"
elif selected == "VGG16":
    st.title("Навчання VGG16")
    
    # Створюємо інпут для введення кількості епох навчання (обмежений діапазоном)
    epochsSetVGG16 = st.number_input(label = f'Введіть кількість епох навчання VGG16(2..{maxEpochSetVGG16})', min_value = 2, max_value = maxEpochSetVGG16, value = epochsSetVGG16, step = 1)
    
    # Створюємо кнопку для початку тренування моделі VGG16
    traineVGG16 = st.button('Тренувати модель VGG16', disabled = traineCNN)

    print("epochsSetVGG16 =", epochsSetVGG16)
    print("traineVGG16 =", traineVGG16)

    # Якщо кнопка тренування натиснута
    if traineVGG16:
        st.title('Триває навчання моделі...')
        
        # Створюємо progress bar для відображення прогресу тренування
        progressVGG16Bar = st.progress(progressCalcVGG16, text= f"Прогрес навчання {progressCalcVGG16}%")

        # Завантажуємо та підготовлюємо дані для VGG16 моделі
        train_imagesVGG16, train_labels, test_imagesVGG16, test_labels = datasetDownload(0)  # 0 - означає VGG16
        
        # Створюємо модель VGG16
        modelVGG16 = createVGG16Model()
        
        # Додаємо callback для експоненційного зменшення швидкості навчання
        lr_schedulerVGG16 = tf.keras.callbacks.LearningRateScheduler(exponential_decayVGG16)
        
        # Навчаємо модель
        historyVGG16 = modelVGG16.fit(train_imagesVGG16, train_labels, batch_size=batchSetVGG16, epochs=epochsSetVGG16, validation_data=(test_imagesVGG16, test_labels), callbacks=[lr_schedulerVGG16], verbose=0)
        
        st.success('Навчання завершено :) ')
        
        # Візуалізуємо графік точності та втрат
        st.pyplot(showPlot(historyVGG16))
        
        # Очищаємо графік для наступного використання
        plotsClear()

        # Оцінюємо модель
        lossVGG16, accuracyVGG16 = modelVGG16.evaluate(test_imagesVGG16, test_labels, verbose=0)
        st.write(f"Точність розпізнавання: {accuracyVGG16:.2f}")

        # Зберігаємо модель у файл
        pathVGG16Model = os.path.join(current_dir, "saveModels\\", "VGG16Model.keras")
        modelVGG16.save(pathVGG16Model)

        st.title(f"Модель збережена за шляхом:")
        st.write(pathVGG16Model)

        st.title('Тестування на датасеті')

        # Вибираємо випадкові зразки для тестування
        random_choice = random.randint(0, len(test_labels) - showDatasetSamples)
        test_images = test_imagesVGG16[random_choice:random_choice + showDatasetSamples]
        test_labels = test_labels[random_choice:random_choice + showDatasetSamples]
        
        # Прогнозуємо результати для зразків
        predicted_results = modelVGG16.predict(test_images)
        predicted_labels = tf.argmax(predicted_results, 1).numpy()

        # Візьмемо картинки, які не перетворені у 32*32*3 для зручності відображення
        _, _, test_images_, test_labels_ = datasetDownload(1)  

        for sampleVGG16 in range(showDatasetSamples):
            plt.imshow(test_images_[random_choice + sampleVGG16].reshape(28, 28), cmap='gray')
            st.write(f"Справжня мітка: {classes[test_labels_[random_choice + sampleVGG16]]}, Прогнозована мітка: {classes[predicted_labels[sampleVGG16]]}")
            st.pyplot(plt)
            plotsClear()
            st.write("Графік прогнозів для міток")
            st.pyplot(lablesPlot(predicted_results[sampleVGG16]))
            print("predicted_results[sampleVGG16]", predicted_results[sampleVGG16])
            # Очищаємо графік для наступного використання
            plotsClear()
  
        traineVGG16 = False

# Якщо вибрано опцію "Models test"
elif selected == "Models test":
    st.title("Завантаження зображень для тестування")

    # Завантажуємо зображення користувача
    uploaded_file = st.file_uploader("Виберіть зображення...", type=["jpg", "jpeg", "png", "gif"])

    if uploaded_file is not None:
        # Відкриваємо завантажене зображення за допомогою бібліотеки Pillow
        uploadImage = Image.open(uploaded_file)
        
        # Змінюємо розмір зображення та перетворюємо його у чорно-білий формат з автоматичним вибором контрасту
        resizedImage = uploadImage.resize((28, 28))
        grayscaleImage = ImageOps.grayscale(resizedImage)
        autocontrastImage = ImageOps.autocontrast(grayscaleImage, cutoff=30)

        # Відображаємо оброблене зображення у веб-додатку Streamlit
        st.image(autocontrastImage, caption='Оброблене зображення')

        # Створюємо випадаючий список для вибору моделі (CNN або VGG16)
        selectModel = st.selectbox('Виберіть модель', ["CNN","VGG16"])
        
        # Створюємо кнопку для початку розпізнавання зображення
        checkPictureEnable = st.button('Розпізнати зображення')
        
        predicted_label = None
        predicted_result = []

        print("selectModel =", selectModel)  # Виводимо вибрану модель у консоль для дебагу
        print("checkPictureEnagle =", checkPictureEnable)  # Виводимо стан кнопки у консоль для дебагу

        if checkPictureEnable:  # Якщо кнопка натиснута
            checkPictureEnable = False  # Відключаємо кнопку, щоб вона не спрацювала ще раз
            imageToArray = np.array(autocontrastImage)  # Перетворюємо зображення у масив numpy

            if selectModel == "CNN":  # Якщо вибрано CNN модель
                modelLoadPath = os.path.join(current_dir, "saveModels\\", "CNNModel.keras")
                lodaCNNModel = keras.models.load_model(modelLoadPath)  # Завантажуємо збережену модель CNN

                imageToCNN = imageToArray.reshape((-1, 28, 28, 1))  # Підготуємо зображення для CNN моделі
                
                predicted_result = lodaCNNModel.predict(imageToCNN)  # Робимо прогноз з використанням моделі
                predicted_label = tf.argmax(predicted_result, 1).numpy()  # Отримуємо мітку передбаченого класу
            else:  # Якщо вибрано VGG16 модель
                modelLoadPath = os.path.join(current_dir, "saveModels\\", "VGG16Model.keras")
                lodaVGG16Load = keras.models.load_model(modelLoadPath)  # Завантажуємо збережену модель VGG16

                imageToArray = np.reshape(imageToArray, (1, 28, 28))  # Підготуємо зображення для VGG16 моделі
                imageToVGG16 = imageToArray.astype('float32')
                imageToVGG16 = np.repeat(imageToVGG16[..., np.newaxis], 3, -1)  # Додаємо RGB канал
                imageToVGG16 = tf.image.resize(imageToVGG16, (32, 32))  # Змінюємо розмір зображення до 32x32

                predicted_result = lodaVGG16Load.predict(imageToVGG16)  # Робимо прогноз з використанням моделі
                predicted_label = tf.argmax(predicted_result, 1).numpy()  # Отримуємо мітку передбаченого класу
    
            plt.imshow(autocontrastImage)  # Відображаємо оброблене зображення
            st.write(f"Прогнозований клас: {classes[int(predicted_label)]}")  # Виводимо передбачений клас у веб-додатку
            st.pyplot(plt)  # Відображаємо графік зображення
            # Очищаємо графік для наступного використання
            plotsClear() 
            st.title("Гістограма прогнозу класів")
            st.pyplot(lablesPlot(predicted_result[0]))  # 0 тому що лиш одна картинка
            # Очищаємо графік для наступного використання
            plotsClear() 

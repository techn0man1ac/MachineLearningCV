# ML Application with Streamlit

![Model training screenshot](https://raw.githubusercontent.com/techn0man1ac/MachineLearningCV/refs/heads/main/screenshots/ModelTraining.png)

This project is a web-based machine learning application that allows training and testing CNN and VGG16 models based on the Fashion MNIST dataset. The application was created using [Streamlit](https://github.com/streamlit/streamlit).

# 📋 Description

The application has the following functions:
1. Training the CNN model on black and white images (Fashion MNIST).
2. Training the VGG16 model on color images (converted from Fashion MNIST).
3. Visualization of training results, including loss and accuracy graphs.
4. Testing of images uploaded by the user based on the selected model.
5. Use pre-trained models in the `Model test` tab(without training). Pre-trained models should be in the `saveModels` directory. Here my prepared models:

https://drive.google.com/drive/folders/11ptCnpoiAlmtYGRsQOJaR54Vhexvcw34?usp=sharing

# 🛠️ File structure

- `main.py` The script to run the Streamlit application.
- `app.py` The main module containing the application functionality, including model definition, training, visualization, and testing.

# Running the application

```bash
python main.py
```

Or directly:

```bash
streamlit run main.py
```

# 📊 Models

## CNN:

Uses 2D convolutional layers.
The architecture includes Dropout to prevent overfitting.
Trained on 28x28 grayscale images.

## VGG16:

Pre-trained on ImageNet.
Input images are scaled to 32x32 with three channels (RGB).
Only the VGG16 main unit (frozen) is used.

# 🖼️ How to work with the application

Runn app, click the `Model Test` tab and select the image you want to recognize. After selecting the image, it will be converted to the correct format and size for submitting the model for recognition. Two models(CNN/VGG16) are trained to recognize type of product(clothing item) such as bags, T-shirts, dresses... Based on a dataset [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).

![Recognize image screenshot](https://raw.githubusercontent.com/techn0man1ac/MachineLearningCV/refs/heads/main/screenshots/RecognizeImage.png)

1. Training models
Select `CNN` or `VGG16` in the menu.
Enter the number of epochs and click “Train model”.
2. Testing
Upload an image in .jpg, .jpeg, .png, or .gif format.
Select a model (CNN or VGG16).
Click `Recognize image`.
3. Saving models
After completing the training of the models app saved traine model in the `saveModels` folder in the `.keras` format.

# 📂 Example result.

## Training graphs

![Traine statistics screenshot](https://raw.githubusercontent.com/techn0man1ac/MachineLearningCV/refs/heads/main/screenshots/TraineStatistics.png)

Graph of loss and accuracy.

## Image recognition

![Shirt image recognition screenshot](https://raw.githubusercontent.com/techn0man1ac/MachineLearningCV/refs/heads/main/screenshots/Shirt.png)

Image with class prediction and probability histogram.

![Lable histogram](https://raw.githubusercontent.com/techn0man1ac/MachineLearningCV/refs/heads/main/screenshots/LableHistogram.png)

## 📑 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/techn0man1ac/MachineLearningCV/blob/main/LICENSE) file for details. 

Streamlit software is also distributed under the [Apache-2.0 license](https://github.com/streamlit/streamlit?tab=Apache-2.0-1-ov-file).

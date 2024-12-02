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

# CNN Model Parameters
epochsSetCNN = 5  # Number of training epochs for the CNN model
maxEpochSetCNN = 100  # Maximum allowable number of epochs for CNN training
batchSetCNN = 512  # Batch size for CNN training
learningRateSetCNN = 0.00045  # Initial learning rate for CNN training
decayRateSetCNN = 0.0015  # Learning rate decay factor for CNN training

# VGG16 Model Parameters
epochsSetVGG16 = 5  # Number of training epochs for the VGG16 model
maxEpochSetVGG16 = 100  # Maximum allowable number of epochs for VGG16 training
batchSetVGG16 = 512  # Batch size for VGG16 training
learningRateSetVGG16 = 0.01  # Initial learning rate for VGG16 training
decayRateSetVGG16 = 0.001  # Learning rate decay factor for VGG16 training

# General Configuration
showDatasetSamples = 10  # Number of samples to display for verification after training

# State Management
traineCNN = False  # Indicates whether the CNN model is currently being trained
traineVGG16 = False  # Indicates whether the VGG16 model is currently being trained

# Logical flags
progressCalcCNN = 0 # Logical flag to determine whether to train the CNN model
progressCalcVGG16 = 0 # Logical flag to determine whether to train the VGG16 model

# Current working directory
current_dir = os.getcwd()

# List of dataset class names
classes = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

# Function to load the dataset. Takes isCNN argument that determines if this is a CNN model.
def datasetDownload(isCNN):
    # Load the fashion_mnist dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
    print("Dataset is downloaded now.")

    if(isCNN == True):
        # Expand the dimensions of the images to add a channel (for convolutional layers)
        train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
        test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    else:
        # Change data type to float32 to increase computation speed
        train_images = train_images.astype('float32')
        test_images = test_images.astype('float32')

        # Resize images to (32, 32, 3) to match VGG16
        train_images = np.repeat(train_images[..., np.newaxis], 3, -1)  # RGB channel
        test_images = np.repeat(test_images[..., np.newaxis], 3, -1)
        train_images = tf.image.resize(train_images, (32, 32))
        test_images = tf.image.resize(test_images, (32, 32))

        # Convert labels to one-hot encoded vectors (10 for fashion_mnist)
        train_labels = to_categorical(train_labels, 10)
        test_labels = to_categorical(test_labels, 10)
    
    return train_images, train_labels, test_images, test_labels

# Function to create a CNN model
def createCNNModel():
    # Create an enhanced neural network with convolutional layers
    modelCNN = models.Sequential([
        # First convolutional layer
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Flatten the feature vector before applying to fully connected layers
        layers.Flatten(),
        
        # Add a fully connected layer with Dropout to prevent overfitting
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='linear'),
        layers.Dropout(0.6),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    modelCNN.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    
    return modelCNN

# Function to create a VGG16 model
def createVGG16Model():
    conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
    conv_base.trainable = False

    # Create an enhanced neural network with convolutional layers
    modelVGG16 = models.Sequential([
        conv_base,
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        # Output layer with 10 classes
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    modelVGG16.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=learningRateSetVGG16),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return modelVGG16

# Exponential learning rate decay for the CNN model
def exponential_decayCNN(epoch, lr):
    global progressCalcCNN
    progressCalcCNN = int(((epoch + 1) / epochsSetCNN) * 100)
    progressCNNBar.progress(progressCalcCNN, text= f"Training progress {progressCalcCNN}%")  # Update progress bar

    if(epoch > 0):  
        decay_rate = decayRateSetCNN
        return float(lr * tf.math.exp(-decay_rate * epoch))
    else:
        return learningRateSetCNN   

# Exponential learning rate decay for the VGG16 model
def exponential_decayVGG16(epoch, lr):
    global progressCalcVGG16
    progressCalcVGG16 = int(((epoch + 1) / epochsSetVGG16) * 100)
    progressVGG16Bar.progress(progressCalcVGG16, text= f"Training progress {progressCalcVGG16}%")  # Update progress bar

    if(epoch > 0):    
        decay_rate = decayRateSetVGG16
        return float(lr * tf.math.exp(-decay_rate * epoch))
    else:
        return learningRateSetVGG16   

# Function to visualize accuracy and loss plots
def showPlot(traineHistory):
    # Visualize loss and accuracy plots
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(traineHistory.history['loss'], label='Loss on training data')
    plt.plot(traineHistory.history['val_loss'], label='Loss on validation data')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(traineHistory.history['accuracy'], label='Accuracy on training data')
    plt.plot(traineHistory.history['val_accuracy'], label='Accuracy on validation data')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    return plt

def lablesPlot(predictedLablees):
    # Ensure this is a one-dimensional array
    predictedLablees_ = np.squeeze(np.array(predictedLablees))

    # Index of the class with the highest probability
    predictedClasses = np.argmax(predictedLablees_)

    # Dynamic y-axis scaling
    y_min = max(0, min(predictedLablees_))  # Padding down (but not below 0)
    y_max = min(1.1, max(predictedLablees_))  # Padding up (but not above 1.1)

    # Create the plot
    plt.figure(figsize=(10, 5))
    bars = plt.bar(classes, predictedLablees, color='gray')

    # Highlight the predicted class with color
    bars[predictedClasses].set_color('orange')

    # Additional styles
    plt.title(f'Predicted class: {classes[predictedClasses]}', fontsize=16, color='blue')
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(y_min, y_max)  # Apply automatic scaling
    plt.tight_layout()
    return plt

# Clear the plot for further use
def plotsClear():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

# Set the title for the web app
st.title('HW 16')

# Create a navigation menu using streamlit_option_menu
selected = option_menu(menu_title=None, options=["CNN", "VGG16", "Models test"],
                       default_index=2,
                       orientation="horizontal")

# If the "CNN" option is selected
if selected == "CNN":
    st.title("CNN Training")
    
    # Create an input field to specify the number of training epochs (restricted range)
    epochsSetCNN = st.number_input(label=f'Enter the number of CNN training epochs (2..{maxEpochSetCNN})', min_value=2, max_value=maxEpochSetCNN, value=epochsSetCNN, step=1)
    
    # Create a button to start CNN model training
    traineCNN = st.button('Train CNN model', disabled=traineVGG16)

    print("epochsSetCNN =", epochsSetCNN)
    print("traineCNN =", traineCNN)

    # If the training button is clicked
    if traineCNN:
        st.title('Training the model...')
        
        # Create a progress bar to display training progress
        progressCNNBar = st.progress(progressCalcCNN, text=f"Training progress {progressCalcCNN}%")

        # Load and prepare data for the CNN model
        train_imagesCNN, train_labels, test_imagesCNN, test_labels = datasetDownload(1)  # 1 indicates CNN
        
        # Create the CNN model
        modelCNN = createCNNModel()
        
        # Add a callback for exponential learning rate decay
        lr_schedulerCNN = tf.keras.callbacks.LearningRateScheduler(exponential_decayCNN)
        
        # Train the model
        historyCNN = modelCNN.fit(train_imagesCNN, train_labels, batch_size=batchSetCNN, epochs=epochsSetCNN, validation_data=(test_imagesCNN, test_labels), callbacks=[lr_schedulerCNN], verbose=0)
        
        st.success('Training completed :)')
        
        # Visualize accuracy and loss plots
        st.pyplot(showPlot(historyCNN))
        
        # Clear the plot for future use
        plotsClear()

        # Evaluate the model
        lossCNN, accuracyCNN = modelCNN.evaluate(test_imagesCNN, test_labels, verbose=0)
        st.write(f"Recognition accuracy: {accuracyCNN:.2f}")

        # Save the model to a file
        pathCNNModel = os.path.join(current_dir, "saveModels", "CNNModel.keras")
        modelCNN.save(pathCNNModel)

        st.title(f"Model saved at path:")
        st.write(pathCNNModel)

        st.title('Dataset Testing')

        # Select random samples for testing
        randomChoice = random.randint(0, len(test_labels) - showDatasetSamples)
        test_images = test_imagesCNN[randomChoice:randomChoice + showDatasetSamples]
        test_labels = test_labels[randomChoice:randomChoice + showDatasetSamples]
        
        # Predict results for the samples
        predicted_results = modelCNN.predict(test_images)
        predicted_labels = tf.argmax(predicted_results, 1).numpy()

        for sampleCNN in range(showDatasetSamples):
            plt.imshow(test_images[sampleCNN].reshape(28, 28), cmap='gray')
            st.write(f"True label: {classes[test_labels[sampleCNN]]}, Predicted label: {classes[predicted_labels[sampleCNN]]}")
            st.pyplot(plt)
            # Clear the plot for future use
            plotsClear()
            st.write("Prediction chart for labels")
            st.pyplot(lablesPlot(predicted_results[sampleCNN]))
            print("predicted_results[sampleCNN]", predicted_results[sampleCNN])
            # Clear the plot for future use
            plotsClear()

        traineCNN = False
        
# If the "VGG16" option is selected
elif selected == "VGG16":
    st.title("VGG16 Training")
    
    # Create an input field to specify the number of training epochs (restricted range)
    epochsSetVGG16 = st.number_input(label=f'Enter the number of VGG16 training epochs (2..{maxEpochSetVGG16})', min_value=2, max_value=maxEpochSetVGG16, value=epochsSetVGG16, step=1)
    
    # Create a button to start VGG16 model training
    traineVGG16 = st.button('Train VGG16 model', disabled=traineCNN)

    print("epochsSetVGG16 =", epochsSetVGG16)
    print("traineVGG16 =", traineVGG16)

    # If the training button is clicked
    if traineVGG16:
        st.title('Training the model...')
        
        # Create a progress bar to display training progress
        progressVGG16Bar = st.progress(progressCalcVGG16, text=f"Training progress {progressCalcVGG16}%")

        # Load and prepare data for the VGG16 model
        train_imagesVGG16, train_labels, test_imagesVGG16, test_labels = datasetDownload(0)  # 0 indicates VGG16
        
        # Create the VGG16 model
        modelVGG16 = createVGG16Model()
        
        # Add a callback for exponential learning rate decay
        lr_schedulerVGG16 = tf.keras.callbacks.LearningRateScheduler(exponential_decayVGG16)
        
        # Train the model
        historyVGG16 = modelVGG16.fit(train_imagesVGG16, train_labels, batch_size=batchSetVGG16, epochs=epochsSetVGG16, validation_data=(test_imagesVGG16, test_labels), callbacks=[lr_schedulerVGG16], verbose=0)
        
        st.success('Training completed :)')
        
        # Visualize accuracy and loss plots
        st.pyplot(showPlot(historyVGG16))
        
        # Clear the plot for future use
        plotsClear()

        # Evaluate the model
        lossVGG16, accuracyVGG16 = modelVGG16.evaluate(test_imagesVGG16, test_labels, verbose=0)
        st.write(f"Recognition accuracy: {accuracyVGG16:.2f}")

        # Save the model to a file
        pathVGG16Model = os.path.join(current_dir, "saveModels", "VGG16Model.keras")
        modelVGG16.save(pathVGG16Model)

        st.title(f"Model saved at path:")
        st.write(pathVGG16Model)

        st.title('Dataset Testing')

        # Select random samples for testing
        random_choice = random.randint(0, len(test_labels) - showDatasetSamples)
        test_images = test_imagesVGG16[random_choice:random_choice + showDatasetSamples]
        test_labels = test_labels[random_choice:random_choice + showDatasetSamples]
        
        # Predict results for the samples
        predicted_results = modelVGG16.predict(test_images)
        predicted_labels = tf.argmax(predicted_results, 1).numpy()

        # Use the images that are not resized to 32x32x3 for better visualization
        _, _, test_images_, test_labels_ = datasetDownload(1)  

        for sampleVGG16 in range(showDatasetSamples):
            plt.imshow(test_images_[random_choice + sampleVGG16].reshape(28, 28), cmap='gray')
            st.write(f"True label: {classes[test_labels_[random_choice + sampleVGG16]]}, Predicted label: {classes[predicted_labels[sampleVGG16]]}")
            st.pyplot(plt)
            plotsClear()
            st.write("Prediction chart for labels")
            st.pyplot(lablesPlot(predicted_results[sampleVGG16]))
            print("predicted_results[sampleVGG16]", predicted_results[sampleVGG16])
            # Clear the plot for future use
            plotsClear()
  
        traineVGG16 = False

# If the "Models test" option is selected
elif selected == "Models test":
    st.title("Upload Images for Testing")

    # Upload an image provided by the user
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "gif"])

    if uploaded_file is not None:
        # Open the uploaded image using the Pillow library
        uploadImage = Image.open(uploaded_file)
        
        # Resize the image and convert it to grayscale with auto-contrast adjustment
        resizedImage = uploadImage.resize((28, 28))
        grayscaleImage = ImageOps.grayscale(resizedImage)
        autocontrastImage = ImageOps.autocontrast(grayscaleImage, cutoff=30)

        # Display the processed image in the Streamlit web app
        st.image(autocontrastImage, caption='Processed Image')

        # Create a dropdown menu to select a model (CNN or VGG16)
        selectModel = st.selectbox('Choose a model', ["CNN", "VGG16"])
        
        # Create a button to start image recognition
        checkPictureEnable = st.button('Recognize Image')
        
        predicted_label = None
        predicted_result = []

        print("selectModel =", selectModel)  # Output the selected model to the console for debugging
        print("checkPictureEnable =", checkPictureEnable)  # Output the button state to the console for debugging

        if checkPictureEnable:  # If the button is clicked
            checkPictureEnable = False  # Disable the button to prevent multiple triggers
            imageToArray = np.array(autocontrastImage)  # Convert the image to a numpy array

            if selectModel == "CNN":  # If the CNN model is selected
                modelLoadPath = os.path.join(current_dir, "saveModels", "CNNModel.keras")
                lodaCNNModel = keras.models.load_model(modelLoadPath)  # Load the saved CNN model

                imageToCNN = imageToArray.reshape((-1, 28, 28, 1))  # Prepare the image for the CNN model
                
                predicted_result = lodaCNNModel.predict(imageToCNN)  # Make a prediction using the model
                predicted_label = tf.argmax(predicted_result, 1).numpy()  # Get the predicted class label
            else:  # If the VGG16 model is selected
                modelLoadPath = os.path.join(current_dir, "saveModels", "VGG16Model.keras")
                lodaVGG16Load = keras.models.load_model(modelLoadPath)  # Load the saved VGG16 model

                imageToArray = np.reshape(imageToArray, (1, 28, 28))  # Prepare the image for the VGG16 model
                imageToVGG16 = imageToArray.astype('float32')
                imageToVGG16 = np.repeat(imageToVGG16[..., np.newaxis], 3, -1)  # Add the RGB channel
                imageToVGG16 = tf.image.resize(imageToVGG16, (32, 32))  # Resize the image to 32x32

                predicted_result = lodaVGG16Load.predict(imageToVGG16)  # Make a prediction using the model
                predicted_label = tf.argmax(predicted_result, 1).numpy()  # Get the predicted class label
    
            plt.imshow(autocontrastImage)  # Display the processed image
            st.write(f"Predicted class: {classes[int(predicted_label)]}")  # Output the predicted class in the web app
            st.pyplot(plt)  # Display the image plot
            # Clear the plot for future use
            plotsClear() 
            st.title("Class Prediction Histogram")
            st.pyplot(lablesPlot(predicted_result[0]))  # 0 because there's only one image
            # Clear the plot for future use
            plotsClear()

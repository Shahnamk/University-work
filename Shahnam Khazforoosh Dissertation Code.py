#Imported Modules
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
import seaborn as sns
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
from keras.applications.xception import Xception
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, load_model, Sequential
from keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam
from keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

print("Running data cleaning process please wait")

#Importing Dataset from computer (Please only edit the importation of the dataset)
image = glob.glob('C:/Users/Joe/OneDrive/Desktop/Kaggleimages/**/*.png', recursive=True)
    

#Creating arrays to store labels and images
images = []
labels = []
#Pre-processing the dataset to be resized and stored into the correct arrays
for i in image[0:20000]:
    image = cv2.imread(i)
    if i.endswith('.png'):
        #Checking which label the image belongs to
        label=i[-5]
        #Resizing the images to an 100x100 dimension
        resized_image = (cv2.resize(image, (100,100), interpolation=cv2.INTER_CUBIC))
        #Storing images into array
        images.append(resized_image)
        #Storing labels into array
        if label == "0":
            labels.append(0)
        elif label == "1":
            labels.append(1)

#Number of cancerous and non-cancerous images from the 20,000 image datatset
print(labels.count(0), "total number of Non-cancerous Images")
print(labels.count(1), "total number of Cancerous Images")

#Converting images array into a NumPy array and normalizing the data            
images = np.array(images)
images = images.astype(np.float32)
images = images/255

#Creating train and test sample from dataset
X_train,X_test,Y_train,Y_test=train_test_split(images,labels,test_size=0.2)

#Making the labels categorical to work with the CNN's configurations
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

#Ensuring the test and train data has been redeveloped into an array
Y_train = np.array(Y_train)
X_train = np.array(X_train)
Y_test = np.array(Y_test)
X_test = np.array(X_test)


#Setting up the ability to choose between all CNN's (Please note I did not add the ability to run tests
#one after another, as issues with PC RAM will gradually decrease result quality and breaks the code in Python)
choice = int(input("Hello, please type 1 for VGG16, 2 for ResNet50, 3 for Xception, 4 for self-made CNN or 0 to terminate the code "))

while choice <0 or choice >4:
    state = True
    #If the user types an incorrect value, the programme will ask for a new value until a correct one is typed
    while state == True:
        print("Input was", choice)
        choice = int(input("Incorrect value has been typed please type 1 for VGG16, 2 for ResNet50, 3 for Xception, 4 for self-made CNN or 0 to terminate the code "))
        if choice <0 or choice >4:
            state = True
        else:
            state = False
            
else:
    #Terminating the code     
    if choice == 0:
        print("code has now been terminated")
    #This CNN was chosen
    if choice == 1:
        
        print("VGG16 has been chosen for analysis")
    
        #Loading VGG16 and training its weights using the ImageNet Dataset   
        model = VGG16(include_top=False, weights='imagenet', input_shape=(100,100,3),pooling='avg')
        
        #Ensuring training of new weights does not occur
        for layer in model.layers:
          layer.trainable = False
          
        #Ensuring the model can output data that works with the Breast Cancer Histopathology dataset for
        #its final hidden layer
        model_outputs=model.output
        model_outputs = Flatten()(model.output)
        
        preds=Dense(2, activation='softmax')(model_outputs)
        
        model=Model(inputs=model.input,outputs=preds)
        model.summary()
        
        #Setting optimizer, learning rate and loss for model
        model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
        
        #Creating variations/edits of the same train datatset for each epoch  
        Imagedata = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=180,
            horizontal_flip=True,vertical_flip = True)
        
        #Creating an early stopper in case no improvements based on validation loss occur
        Stopper = EarlyStopping(monitor='val_loss', patience=3, mode='min')
        
        #Saving the optimal Model dependent on its lowest Validation Loss Value
        Optimal_find = ModelCheckpoint('Optimal_model.h5', monitor='val_loss', mode='min', verbose=1, 
                                           save_best_only=True)
        
        #Fitting the model with the following configurations
        history = model.fit(Imagedata.flow(X_train,Y_train,batch_size=64),steps_per_epoch=len(X_train)/64, 
                            epochs=30,validation_data=(X_test, Y_test), verbose=1, 
                            callbacks=[Stopper, Optimal_find])
        
        
        #Plotting the validation, training accuracy and loss over epochs
        fig1 = plt.gcf()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.axis(ymin=0.2,ymax=1)
        plt.grid()
        plt.title('Model Accuracy VGG16')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['train', 'validation'])
        plt.show()
        
        fig1 = plt.gcf()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.axis(ymin=0.2,ymax=1)
        plt.grid()
        plt.title('Model Loss VGG16')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(['train', 'validation'])
        plt.show()
        
        #Loading the optimal model to be applied on the test dataset and predict its outputs
        model = load_model('Optimal_model.h5')
        
        y_pred = model.predict(X_test)
        y_pred_labels = np.argmax(y_pred, axis = 1)
        y_true_labels = np.argmax(Y_test,axis=1)
        classes = ['Not cancerous','Cancerous']
        
        #Creating Confusion Matrix
        confusion = (confusion_matrix(y_true=y_true_labels, y_pred=y_pred_labels))
        print("Confusion Matrix")
        print(confusion)
        
        #Creating Classification report
        print('Classification Report')
        print(classification_report(y_true=y_true_labels, y_pred=y_pred_labels))
        
        #Creating heatmap of Confusion Matrix
        heatmap = sns.heatmap(confusion/np.sum(confusion), annot=True,xticklabels=classes,yticklabels=classes,
                              fmt='.2%', cmap='Blues')
        
        plt.title('VGG16', fontsize = 20)
        plt.xlabel('Predicted', fontsize = 15)
        plt.ylabel('True', fontsize = 15)
        
        plt.show()
        print("VGG16 analysis has been completed, please re-run code to test other CNN's")
        
    elif choice == 2:
    #This CNN was chosen    
        print("ResNet50 has been chosen for analysis")
        
        #Loading ResNet50 and training its weights using the ImageNet Dataset  
        model = ResNet50(include_top=False, weights='imagenet', input_shape=(100,100,3),pooling='avg')
        
        #Ensuring training of new weights does not occur
        for layer in model.layers:
          layer.trainable = False
          
        #Ensuring the model can output data that works with the Breast Cancer Histopathology dataset for
        #its final hidden layer
        model_outputs=model.output
        model_outputs = Flatten()(model.output)
        
        #Final layer
        #additonal layers
        preds=Dense(2, activation='softmax')(model_outputs)
        
        model=Model(inputs=model.input,outputs=preds)
        model.summary()
        
        #Setting optimizer, learning rate and loss for model
        model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
        
        #Creating variations/edits of the same train datatset for each epoch  
        Imagedata = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=180,
            horizontal_flip=True,vertical_flip = True)
        
        #Creating an early stopper in case no improvements based on validation loss occur        
        Stopper = EarlyStopping(monitor='val_loss', patience=3, mode='min')
        
        #Saving the optimal Model dependent on its lowest Validation Loss Value        
        Optimal_find = ModelCheckpoint('Optimal_model.h5', monitor='val_loss', mode='min', verbose=1, 
                                           save_best_only=True)
      
        #Fitting the model with the following configurations        
        history = model.fit(Imagedata.flow(X_train,Y_train,batch_size=64),steps_per_epoch=len(X_train)/64, 
                            epochs=30,validation_data=(X_test, Y_test), verbose=1, 
                            callbacks=[Stopper, Optimal_find])
        
        
        #Plotting the validation, training accuracy and loss over epochs
        fig1 = plt.gcf()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.axis(ymin=0.2,ymax=1)
        plt.grid()
        plt.title('Model Accuracy ResNet50')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['train', 'validation'])
        plt.show()
        
        fig1 = plt.gcf()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.axis(ymin=0.2,ymax=1)
        plt.grid()
        plt.title('Model Loss ResNet50')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(['train', 'validation'])
        plt.show()
        
        #Loading the optimal model to be applied on the test dataset and predict its outputs
        model = load_model('Optimal_model.h5')
        
        y_pred = model.predict(X_test)
        y_pred_labels = np.argmax(y_pred, axis = 1)
        y_true_labels = np.argmax(Y_test,axis=1)
        classes = ['Not cancerous','Cancerous']
        
        #Creating Confusion Matrix
        confusion = (confusion_matrix(y_true=y_true_labels, y_pred=y_pred_labels))
        print("Confusion Matrix")
        print(confusion)
        
        #Creating Classification report
        print('Classification Report')
        print(classification_report(y_true=y_true_labels, y_pred=y_pred_labels))
        
        #Creating heatmap of Confusion Matrix
        heatmap = sns.heatmap(confusion/np.sum(confusion), annot=True,xticklabels=classes,yticklabels=classes,
                              fmt='.2%', cmap='Blues')
        
        plt.title('ResNet50', fontsize = 20)
        plt.xlabel('Predicted', fontsize = 15)
        plt.ylabel('True', fontsize = 15)
        
        plt.show()
        
        print("ResNet50 analysis has been completed, please re-run code to test other CNN's")
    
    elif choice == 3:
    #This CNN was chosen        
        print("Xception has been chosen for analysis")
        
        #Loading Xception and training its weights using the ImageNet Dataset          
        model = Xception(include_top=False, weights='imagenet', input_shape=(100,100,3),pooling='avg')
        
        #Ensuring training of new weights does not occur
        for layer in model.layers:
          layer.trainable = False
          
        #Ensuring the model can output data that works with the Breast Cancer Histopathology dataset for
        #its final hidden layer
        model_outputs=model.output
        model_outputs = Flatten()(model.output)
        
        #Final layer
        #additonal layers
        preds=Dense(2, activation='softmax')(model_outputs)
        
        model=Model(inputs=model.input,outputs=preds)
        model.summary()
        
        #Setting optimizer, learning rate and loss for model
        model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

        #Creating variations/edits of the same train datatset for each epoch     
        Imagedata = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=180,
            horizontal_flip=True,vertical_flip = True)
        
        #Creating an early stopper in case no improvements based on validation loss occur    
        Stopper = EarlyStopping(monitor='val_loss', patience=3, mode='min')

        #Saving the optimal Model dependent on its lowest Validation Loss Value          
        Optimal_find = ModelCheckpoint('Optimal_model.h5', monitor='val_loss', mode='min', verbose=1, 
                                           save_best_only=True)
        
        #Fitting the model with the following configurations           
        history = model.fit(Imagedata.flow(X_train,Y_train,batch_size=64),steps_per_epoch=len(X_train)/64, 
                            epochs=30,validation_data=(X_test, Y_test), verbose=1, 
                            callbacks=[Stopper, Optimal_find])
        
        #Plotting the validation, training accuracy and loss over epochs
        fig1 = plt.gcf()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.axis(ymin=0.2,ymax=1)
        plt.grid()
        plt.title('Model Accuracy Xception')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['train', 'validation'])
        plt.show()
        
        fig1 = plt.gcf()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.axis(ymin=0.2,ymax=1)
        plt.grid()
        plt.title('Model Loss Xception')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(['train', 'validation'])
        plt.show()
        
        #Loading the optimal model to be applied on the test dataset and predict its outputs
        model = load_model('Optimal_model.h5')
        
        y_pred = model.predict(X_test)
        y_pred_labels = np.argmax(y_pred, axis = 1)
        y_true_labels = np.argmax(Y_test,axis=1)
        classes = ['Not cancerous','Cancerous']
        
        #Creating Confusion Matrix
        confusion = (confusion_matrix(y_true=y_true_labels, y_pred=y_pred_labels))
        print("Confusion Matrix")
        print(confusion)
        
        #Creating Classification report
        print('Classification Report')
        print(classification_report(y_true=y_true_labels, y_pred=y_pred_labels))
        
        #Creating heatmap of Confusion Matrix
        heatmap = sns.heatmap(confusion/np.sum(confusion), annot=True,xticklabels=classes,yticklabels=classes,
                              fmt='.2%', cmap='Blues')
        
        plt.title('Xception', fontsize = 20)
        plt.xlabel('Predicted', fontsize = 15)
        plt.ylabel('True', fontsize = 15)
        
        plt.show()
        
        print("Xception analysis has been completed, please re-run code to test other CNN's")


    elif choice == 4:
    #This CNN was chosen    
        print("Self-mae CNN has been chosen for analysis")
        
        #Structure of Self-made CNN                
        model = Sequential()
        model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (100, 100, 3), padding="valid", kernel_regularizer=l1(0.001)))
        model.add(Conv2D(32, (3,3), activation ='relu', padding="valid", kernel_regularizer=l1(0.001)))
        model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))
        
        model.add(Conv2D(64, (3,3), activation = 'relu', padding="same"))
        model.add(Conv2D(64, (3,3), activation ='relu', padding="same"))
        model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))
        model.add(Dropout(0.2))
        
        model.add(Conv2D(128, (3,3), activation ='relu', padding="same"))
        model.add(Conv2D(128, (3,3), activation ='relu', padding="same"))
        model.add(Conv2D(128, (3,3), activation ='relu', padding="same"))
        model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))
        model.add(Dropout(0.3))
                
        model.add(Conv2D(256, (3,3), activation ='relu', padding="same"))
        model.add(Conv2D(256, (3,3), activation ='relu', padding="same"))
        model.add(Conv2D(256, (3,3), activation ='relu', padding="same"))
        model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))
        
        model.add(Flatten())
        model.add(Dense(512, activation = "relu"))
        model.add(Dense(512, activation = "relu"))
        model.add(Dense(2, activation = "softmax"))
        model.summary()
        
        #Learning rate schedular created to slow down training in the later epochs
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.0001,
                    decay_steps=(len(X_train)/64)*10,decay_rate=1,staircase=False)
        
        #Setting optimizer, learning rate and loss for model
        model.compile(optimizer=Adam(lr_schedule),loss='categorical_crossentropy',metrics=['accuracy'])
        
        #Creating variations/edits of the same train datatset for each epoch         
        Imagedata = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=180,
            horizontal_flip=True,vertical_flip = True)
        
        #Creating an early stopper in case no improvements based on validation loss occur        
        Stopper = EarlyStopping(monitor='val_loss', patience=3, mode='min')
        
        #Saving the optimal Model dependent on its lowest Validation Loss Value        
        Optimal_find = ModelCheckpoint('Optimal_model.h5', monitor='val_loss', mode='min', verbose=1, 
                                           save_best_only=True)
        
        #Fitting the model with the following configurations             
        history = model.fit(Imagedata.flow(X_train,Y_train,batch_size=64),steps_per_epoch=len(X_train)/64, 
                            epochs=30,validation_data=(X_test, Y_test), verbose=1, 
                            callbacks=[Stopper, Optimal_find])
        
        
        #Plotting the validation, training accuracy and loss over epochs
        fig1 = plt.gcf()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.axis(ymin=0.2,ymax=1)
        plt.grid()
        plt.title('Model Accuracy Self-Made CNN')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['train', 'validation'])
        plt.show()
        
        fig1 = plt.gcf()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.axis(ymin=0.2,ymax=1)
        plt.grid()
        plt.title('Model Loss Self-Made CNN')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(['train', 'validation'])
        plt.show()
        
        #Loading the optimal model to be applied on the test dataset and predict its outputs
        model = load_model('Optimal_model.h5')
        
        y_pred = model.predict(X_test)
        y_pred_labels = np.argmax(y_pred, axis = 1)
        y_true_labels = np.argmax(Y_test,axis=1)
        classes = ['Not cancerous','Cancerous']
        
        #Creating Confusion Matrix
        confusion = (confusion_matrix(y_true=y_true_labels, y_pred=y_pred_labels))
        print("Confusion Matrix")
        print(confusion)
        
        #Creating Classification report
        print('Classification Report')
        print(classification_report(y_true=y_true_labels, y_pred=y_pred_labels))
        
        #Creating heatmap of Confusion Matrix
        heatmap = sns.heatmap(confusion/np.sum(confusion), annot=True,xticklabels=classes,yticklabels=classes,
                              fmt='.2%', cmap='Blues')
        
        plt.title('Self-made CNN', fontsize = 20)
        plt.xlabel('Predicted', fontsize = 15)
        plt.ylabel('True', fontsize = 15)
        
        plt.show()
        
        print("Self-made CNN analysis has been completed, please re-run code to test other CNN's")
    


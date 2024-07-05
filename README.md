# Indian Sign Language Detection using Deep Learning - A CNN Approach

## PROJECT OUTLINE:

Humans converse through a vast collection of languages, allowing them to exchange information with one another. Sign language on the other hand, involves using visual signs and gestures that are vastly used by the people who have hearing and speaking disabilities.  There are over 300 sign languages over the world, that are actively used. The Indian Sign Language was standardized in 2001 and has been actively used all over the country since then. Having a difference in the mode of communication between the hearing people and the people with hearing losses causes a huge barrier and does not allow for easy conversing between the two communities. In addition the absence of certified interpreters also adds to improper communications and lag of understanding. To remove such barriers, researchers in various fields are actively looking for solutions so as the bridge the gap between the two worlds. The purpose of the project is to eradicate such barriers and lead to active an active communication mode between two parties. The research for developing such a system for the Indian Sign Language isn’t very much advanced. In this project, we consider the third edition of the Indian Sign Language (ISL) system and advance it to detect the alphabetic signs The goal of this project is to develop a deep learning system that helps in image classification using CNN to interpret signs generally used in the ISL.

## METHODOLOGY:

This experiment developed an efficient deep learning model to transform the Indian Sign Language signs for letters to textual format. We adapt an ISL dataset, preprocess them and feed them into a Convolution Neural Network(CNN). This CNN model is specifically designed for image processing. At the initial level of the CNN model, we’ve used a convolutional layer for feature extraction and MaxPooling layers for object recognition tasks. This set of outputs is later fed into a flattening layer, whose main function is to convert the 2-dimensional arrays from pooled feature maps into a long continuous linear vector. Such use of multiple layers within the CNN model helps in improving the accuracy and thus providing perfect interpretations.

### SAMPLE DATASET:
<img height="400" alt="dataset" src="https://github.com/Aashima02/Indian-Sign-Language-Detection-using-Deep-Learning---A-CNN-Approach/assets/93427086/1340ab0b-0418-4a70-82a4-03ca07a6f864">


## REQUIREMENTS:

* A suitable python environment
* Python packages:
    
    * tensorflow
    * keras
    * opencv

The above packages can be manually installed using the pip commands as follows:
python
pip install tensorflow
pip install keras
pip install opencv-python


## PROGRAM:
### cnn.py
```
python
# Part 1 - Building the CNN
#importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras import optimizers
# from tensorflow.keras import metrics
# # from metrics import accuracy
# from .accuracy import accuracy


# Initialing the CNN
classifier = Sequential()
# Step 1 - Convolution Layer 
classifier.add(Convolution2D(32, 3,  3, input_shape = (256, 256, 3), activation = 'relu'))
#step 2 - Pooling
classifier.add(MaxPooling2D(pool_size =(2,2)))
# Adding second convolution layer
classifier.add(Convolution2D(32, 3,  3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))
# 3rd convolution layer
classifier.add(Convolution2D(64, 3,  3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))
# Flattening Layer
classifier.add(Flatten())
# Final Connection
classifier.add(Dense(256, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(26, activation = 'softmax'))

# Compile the model
classifier.compile(
              optimizer = optimizers.SGD(lr = 0.01),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

#Part 2 Fittting the CNN to the image
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'mydata/training_set',
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'mydata/test_set',
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical')

model = classifier.fit_generator(
        training_set,
        steps_per_epoch=800,
        epochs=25,
        validation_data = test_set,
        validation_steps = 6500
      )

'''#Saving the model
import h5py
classifier.save('Trained_model.h5')'''


print(model.history.keys())
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss

plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```

### signdetect.py:
```
python
import cv2
import numpy as np

def nothing(x):
    pass

image_x, image_y = 64,64

from keras.models import load_model
classifier = load_model('Trained_model.h5')

def predictor():
       import numpy as np
       from keras.preprocessing import image
       test_image = image.load_img('1.png', target_size=(64, 64))
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)
       
       if result[0][0] == 1:
              return 'A'
       elif result[0][1] == 1:
              return 'B'
       elif result[0][2] == 1:
              return 'C'
       elif result[0][3] == 1:
              return 'D'
       elif result[0][4] == 1:
              return 'E'
       elif result[0][5] == 1:
              return 'F'
       elif result[0][6] == 1:
              return 'G'
       elif result[0][7] == 1:
              return 'H'
       elif result[0][8] == 1:
              return 'I'
       elif result[0][9] == 1:
              return 'J'
       elif result[0][10] == 1:
              return 'K'
       elif result[0][11] == 1:
              return 'L'
       elif result[0][12] == 1:
              return 'M'
       elif result[0][13] == 1:
              return 'N'
       elif result[0][14] == 1:
              return 'O'
       elif result[0][15] == 1:
              return 'P'
       elif result[0][16] == 1:
              return 'Q'
       elif result[0][17] == 1:
              return 'R'
       elif result[0][18] == 1:
              return 'S'
       elif result[0][19] == 1:
              return 'T'
       elif result[0][20] == 1:
              return 'U'
       elif result[0][21] == 1:
              return 'V'
       elif result[0][22] == 1:
              return 'W'
       elif result[0][23] == 1:
              return 'X'
       elif result[0][24] == 1:
              return 'Y'
       elif result[0][25] == 1:
              return 'Z'
       

       

cam = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

cv2.namedWindow("test")

img_counter = 0

img_text = ''
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    img = cv2.rectangle(frame, (425,100),(625,300), (0,255,0), thickness=2, lineType=8, shift=0)

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    imcrop = img[102:298, 427:623]
    hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    cv2.putText(frame, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
    cv2.imshow("test", frame)
    cv2.imshow("mask", mask)
    
    #if cv2.waitKey(1) == ord('c'):
        
    img_name = "1.png"
    save_img = cv2.resize(mask, (image_x, image_y))
    cv2.imwrite(img_name, save_img)
    print("{} written!".format(img_name))
    img_text = predictor()
        

    if cv2.waitKey(1) == 27:
        break


cam.release()
cv2.destroyAllWindows()

```

## FLOW OF THE PROJECT:


1. The cnn.py file contains the main model of the project, containing multiple layers for feature extraction and object detecction.
2. The code is saved as a trained model in a Hierarchial Data Format(HDF5) file format.
3. Run signdetect.py, whose ultimate goal is to recognise the signs.
4. The opencv package creates a bounding box, where the signs are captured and converted to lower blue and upper blue hsv channels for identification.
5. The image is compared with the dataset and the letters are visualised in textual formats.

<img width="500" alt="flow" src="https://github.com/Aashima02/Indian-Sign-Language-Detection-using-Deep-Learning---A-CNN-Approach/assets/93427086/bb74d302-9184-4e4a-8a18-308cd14993e3">


## CNN MODEL ARCHITECTURE:

![CNN ARCHITETURE](https://github.com/Aashima02/Indian-Sign-Language-Detection-using-Deep-Learning---A-CNN-Approach/assets/93427086/d1383de4-dcf4-49d3-9ced-9676f206e0ec)


## OUTPUT:

### Sample Detection Images:
![g](https://github.com/Aashima02/Indian-Sign-Language-Detection-using-Deep-Learning---A-CNN-Approach/assets/93427086/d9e56c1d-9474-4021-b85e-c1220b0b2937)

![u](https://github.com/Aashima02/Indian-Sign-Language-Detection-using-Deep-Learning---A-CNN-Approach/assets/93427086/60c2a7da-ce67-4622-95ac-127d3f60f22f)

![x](https://github.com/Aashima02/Indian-Sign-Language-Detection-using-Deep-Learning---A-CNN-Approach/assets/93427086/c31450a3-f71c-47f2-90da-00d88f2b6f8a)

### Accuracy Plot:
![accuracy plot](https://github.com/Aashima02/Indian-Sign-Language-Detection-using-Deep-Learning---A-CNN-Approach/assets/93427086/7b9d51fc-92c6-4642-9634-4e39a85834d2)


### Loss Plot:
![loss plot](https://github.com/Aashima02/Indian-Sign-Language-Detection-using-Deep-Learning---A-CNN-Approach/assets/93427086/1607d36c-baf9-4544-a54b-83ba05cd964d)



## RESULT:
In this study, we’ve made use of the third edition of the Indian Sign Language (ISL) - 2021 as the standard dataset. The input for sign detection was fed into a bounding box, whose region was converted as an image recognizable by the camera, and similar to the dataset. With the accuracy of 90% and loss of 28%, this model is able to detect individual signs for alphabets.

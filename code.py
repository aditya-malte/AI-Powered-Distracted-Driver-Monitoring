# -*- coding: utf-8 -*-



from keras.models import Model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.applications.mobilenet import MobileNet

#used transfer learning due to nearly identical and small dataset
#specifically used MobileNet instead of VGG to increase inference speed, as this (driver detection tool) would be used on an RPi, and not a GTX Titan
model = MobileNet(include_top=False, weights='imagenet', pooling=max,input_shape=(224,224,3))
for layer in model.layers:
    layer.trainable = False #kept till the last layers fixed due to high data similarity, can be changed if required
x = model.output
#heavy dropout has been used to overcome overfitting
x = MaxPooling2D()(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = Flatten()(x)
x = BatchNormalization()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(1, activation="sigmoid")(x) #predict whether using mobile or not
model_final= Model(input= model.input, output= predictions)
model_final.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                        #can also try sgd
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True
                                   )

test_datagen = ImageDataGenerator(rescale = 1./255)

#(224,224) because of Mobilenet architecture
training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (224,224),
                                                 batch_size = 8,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('validation',
                                            target_size = (224, 224),
                                            batch_size = 8,
                                            class_mode = 'binary')
model_final.fit_generator(training_set,
                         steps_per_epoch = 32,
                         epochs = 45,
                         validation_data = test_set,
                         validation_steps = 16,
                         )
model_final.save('model.hdf5')
model_final.save_weights('model_weights.hdf5')




import numpy as np
from keras.preprocessing import image



#inference phase
#can also be used for checking test accuracy
test_image = image.load_img('test/img_554.jpg', target_size = (224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict_classes(test_image)
print(model.predict(test_image))
training_set.class_indices
print(result)
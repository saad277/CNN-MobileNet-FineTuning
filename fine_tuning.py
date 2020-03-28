from keras.applications import MobileNet
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,GlobalAveragePooling2D
from keras.layers import Conv2D,ZeroPadding2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop


import os 

#MobileNet works on 224x224 images 

img_rows=224
img_cols=224


#Loading the MobileNet model without the top or Fully Connected Layers

MobileNet=MobileNet(weights="imagenet",include_top="false",
                    input_shape=(224,224,3))


#Here we are freezing the last 4 Layers
#Layers are set to trainable as True by default

for layer in MobileNet.layers:
    layer.trainable=False           #setting all layers to be non-trainable 



#Printing layers

c=0
for i in MobileNet.layers:
    #print(c," ",i.__class__.__name__,"--->",i.trainable)
    c=c+1;


#Making the top Fully Connected Layers

def addTopModelMobileNet(bottom_model,num_classes):

    top_model=bottom_model.output
    #top_model=GlobalAveragePooling2D()(top_model)
    top_model=Dense(1024,activation="relu")(top_model)
    top_model=Dense(1024,activation="relu")(top_model)
    top_model=Dense(512,activation="relu")(top_model)
    top_model=Dense(num_classes,activation="softmax")(top_model)
    return top_model


#Adding top model back to original

num_classes=10

FC_head=addTopModelMobileNet(MobileNet,num_classes)

model=Model(inputs=MobileNet.input,outputs=FC_head)

#print(model.summary())

#Training 

train_data_dir="./monkey_breed/train"
valid_data_dir="./monkey_breed/validation"

#Augmentation
train_dataGen=ImageDataGenerator(rescale=1./255,
                                 rotation_range=45,
                                 width_shift_range=0.3,
                                 height_shift_range=0.3,
                                 horizontal_flip=True,
                                 fill_mode="nearest")


valid_dataGen=ImageDataGenerator(rescale=1./255)

batch_size=32


train_generator=train_dataGen.flow_from_directory(train_data_dir,
                                                  target_size=(224,224),
                                                  batch_size=32,
                                                  class_mode="categorical")


valid_generator=valid_dataGen.flow_from_directory(valid_data_dir,
                                                  target_size=(224,224),
                                                  batch_size=32,
                                                  class_mode="categorical")


model.compile(loss="categorical_crossentropy",optimizer=RMSprop(lr=0.001),metrics=["accuracy"])


nb_train_samples=1097
nb_valid_samples=272

epochs=5
batch_size=16


history=model.fit_generator(train_generator,
                            steps_per_epoch=nb_train_samples//batch_size,
                            epochs=5,
                            validation_data=valid_generator,
                            validation_steps=nb_valid_samples//batch_size)






model.save("model_save.h5",overwrite=True);









    

    


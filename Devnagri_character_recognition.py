from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)

training_set = train_datagen.flow_from_directory('E:\\final\\dhcd\\train',
target_size = (32, 32),
batch_size = 32)

classes = training_set.class_indices
#print(classes)
classifier = load_model('E:\\devnagri\\devnagri_character_model.h5')
classifier.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

testImage = image.load_img('C:\\Users\\DEVANSHI\\Desktop\\Dataset\\gna.png', target_size = (32,32,3))
a = np.resize(testImage, (32,32,3))
testimage = image.img_to_array(a)
print(testimage)
testimage = np.expand_dims(testimage, axis = 0)
print(testimage)


plt.title("Input Image")
plt.imshow(a, cmap=plt.cm.binary)
plt.show()

prediction = classifier.predict_classes(testimage)

for key,value in classes.items():
    if value == prediction:
       print(prediction)
       print("The Image is : ",key)


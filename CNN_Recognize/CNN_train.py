# ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
#               13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
#               25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}
#

from keras import layers
from keras.src.optimizers import RMSprop,Adam
from keras import models
from keras.src.legacy.preprocessing.image import ImageDataGenerator
#
#
# # CNN model
#
# model = models.Sequential()
# model.add(layers.Conv2D(31, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
# model.add(layers.Conv2D(31, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(layers.Dropout(0.25))
#
# model.add(layers.Conv2D(62, (3, 3), padding='same', activation='relu'))
# model.add(layers.Conv2D(62, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(layers.Dropout(0.25))
#
# model.add(layers.Conv2D(62, (3, 3), padding='same', activation='relu'))
# model.add(layers.Conv2D(62, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(layers.Dropout(0.25))
#
# model.add(layers.Flatten())
# model.add(layers.Dense(496, activation='relu'))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(31, activation='softmax'))
#
# optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
#
# model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
#
# # Load model
#
# epochs = 10  # for better result increase the epochs
# batch_size = 20
#
# train_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# train_dir = '/home/minhpn/Desktop/Green_Parking/dataset_vietnam_licenses_plate_train'
# test_dir = '/home/minhpn/Desktop/Green_Parking/dataset_vietnam_license_plate_val'
#
# train_generator = train_datagen.flow_from_directory(train_dir, target_size=(28, 28), batch_size=batch_size, shuffle=True, class_mode='categorical')
#
# validation_generator = test_datagen.flow_from_directory(test_dir, target_size=(28, 28), batch_size=batch_size, class_mode='categorical')
#
# for data_batch, label_batch in train_generator:
#     print(f'Data batch shape: {data_batch.shape}')
#     print(f'Label batch shape: {label_batch.shape}')
#     break
#
# print(train_generator.class_indices)
# print(len(train_generator.class_indices))
#
# ################## TRAIN THE MODEL ##################################
#
# history = model.fit(train_generator, steps_per_epoch=100, epochs=30, validation_data=validation_generator, validation_steps=50)






























model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(12, 28, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
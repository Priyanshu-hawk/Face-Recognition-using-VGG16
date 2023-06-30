import pickle

from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras_vggface import VGGFace

def run_train():
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_genrator = train_datagen.flow_from_directory('faces/train',
                                                    target_size=(224,224),
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True)

    valid_generator = train_datagen.flow_from_directory('faces/test',
                                                        target_size=(224,224),
                                                        color_mode='rgb',
                                                        batch_size=32,
                                                        class_mode='categorical',
                                                        shuffle=True)

    NO_CLASS = len(train_genrator.class_indices.values())

    base_model = VGGFace(include_top=False, model='vgg16', input_shape=(224, 224, 3))

    # print(base_model.summary())
    # print(len(base_model.layers))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    preds = Dense(NO_CLASS, activation='softmax')(x)

    model = Model(inputs = base_model.input, outputs=preds)
    model.summary()
    print(len(model.layers))

    for layer in model.layers[:-5]:
        layer.trainable = False
    for layer in model.layers[-5:]:
        layer.trainable=True

    model.compile(optimizer='RMSProp', loss='categorical_crossentropy', metrics=['accuracy'])

    hist = model.fit(train_genrator, batch_size=1, verbose=1, epochs=5, validation_data=valid_generator)

    model.save('tranfer_lrn_face_cnn.h5')

    class_dict = train_genrator.class_indices
    class_dict = {
        value:key for key, value in class_dict.items()
    }

    print(class_dict)

    with open('face_lable.pkl', 'wb') as f:
        pickle.dump(class_dict, f)

if __name__ == '__main__':
    run_train()
from keras.models import load_model
from keras.models import Model
import pickle
from tensorflow.keras.preprocessing import image
import numpy as np
from keras.applications.mobilenet import preprocess_input
# import cosin
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import random
import os

def get_feature_vector():
    base_model = load_model('./tranfer_lrn_face_cnn.h5')
    feature_layer = 'dense_2'
    model = Model(inputs=base_model.input, outputs=base_model.get_layer(feature_layer).output)
    print(model.summary())


    face_label_filename = 'face_lable.pkl'
    with open(face_label_filename, "rb") as f: 
        class_dictionary = pickle.load(f)
    print(class_dictionary)

    class_list = [value for _, value in class_dictionary.items()]
    print(class_list)

    base_dir = './faces/test/'


    for idx,file in enumerate(class_list):
        img_path = base_dir+file+'/'+random.choice(os.listdir(base_dir+file)) #random select one image from each class test folder
    
        img1 = image.load_img(img_path, target_size=(224,224))
        x1 = image.img_to_array(img1)
        x1 = np.expand_dims(x1, axis=0)
        x1 = preprocess_input(x1)
        features1 = model.predict(x1)
        feature_vector1 = features1.flatten()
        class_dictionary[idx] = feature_vector1

    with open('face_feature_vector.pkl', 'wb') as f:
        pickle.dump(class_dictionary, f)

if __name__ == '__main__':
    get_feature_vector()

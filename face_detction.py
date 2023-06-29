from keras.models import load_model
from keras.models import Model
import pickle
from tensorflow.keras.preprocessing import image
import numpy as np
from keras.applications.mobilenet import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import dlib
from keras_vggface import VGGFace

base_model = load_model('./tranfer_lrn_face_cnn.h5')
# base_model = VGGFace(include_top=False, model='vgg16', input_shape=(224, 224, 3))
feature_layer = 'dense_2'
model = Model(inputs=base_model.input, outputs=base_model.get_layer(feature_layer).output)
print(model.summary())

face_label_filename = 'face_lable.pkl'
with open(face_label_filename, "rb") as f: 
    class_dictionary = pickle.load(f)
# print(class_dictionary)

class_list = [value for _, value in class_dictionary.items()]
print(class_list)

face_feature_vector_filename = 'face_feature_vector.pkl'
with open(face_feature_vector_filename, "rb") as f:
    face_feature_vector = pickle.load(f)
# print(face_feature_vector)

unknown_face_img = cv2.imread('face_add.png')

add_new = False

face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

def _trim_css_to_bounds(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

def _rect_to_css(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()

image_width = 224
image_height = 224

cap = cv2.VideoCapture(0)
print(cap.isOpened())
k = 0 
face_queue = []
while True:
    ret, frame = cap.read()
    faces = []
    if ret:
        for face in face_detector(frame, 1):
            faces.append(_trim_css_to_bounds(_rect_to_css(face.rect), frame.shape))
        
        # print(faces)
        if len(faces) == 1:
            for (y1,x2,y2,x1) in faces:
                x1, y1, x2, y2 = x1-10, y1-10, x2+10, y2+10
                roi = frame[y1:y2, x1:x2]
                # print(roi.shape)
                
                print(roi.shape)
                if roi.shape[0] > 150 and roi.shape[1] > 150:
                    size = (image_width, image_height)
                    resized_img = cv2.resize(roi, size)
                    # cv2.imwrite(f'unknown{k}.jpg', resized_img)
                    k += 1
                    image_array = np.array(resized_img)
                    img = image_array.reshape(1,image_width,image_height,3) 
                    # img = img.astype('float32')

                    img1 = preprocess_input(img)

                    features1 = model.predict(img1)
                    feature_vector1 = features1.flatten()

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    color = (255, 0, 255)
                    stroke = 2
                    

                    cs_score = []
                    for i in face_feature_vector:
                        cosine_similarity_score1 = cosine_similarity(feature_vector1.reshape(1, -1), face_feature_vector[i].reshape(1, -1))
                        cs_score.append(cosine_similarity_score1[0][0])
                    print(cs_score)
                    if max(cs_score) > 0.92:
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
                        cv2.putText(frame, class_list[np.argmax(cs_score)], (x1, y1), font, 1, color, stroke, cv2.LINE_AA)
                        print(class_list[np.argmax(cs_score)], max(cs_score))
                        face_queue.append(np.argmax(cs_score))
                    else:
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
                        cv2.putText(frame, 'unknown', (x1, y1), font, 1, color, stroke, cv2.LINE_AA)
                        print('unknown')
                        face_queue.append(-1)
                        if len(face_queue) > 30 and face_queue[-30:].count(-1) > 25:
                            # check last 30 frame for unknown
                            cv2.imshow('frame', unknown_face_img)
                            cv2.waitKey(5000)
                            add_new = True
                            break
            
        cv2.imshow('frame', frame)
        if add_new or cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

if add_new:
    from make_face_data_webcam import create_faces
    from train import run_train
    from feature_vector_genration import get_feature_vector
    input_name = input('Enter name: ')
    create_faces(input_name)
    print('[+] New face added')
    run_train()
    print('[+] Trainning done')
    get_feature_vector()
    print('[+] Feature vector generated')
    print('[+] pls restart the program to see the changes')
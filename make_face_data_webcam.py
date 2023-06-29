import cv2
import time
import sys
import dlib
import os

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('jay.mp4')

test_per = int(1/0.2)
print(test_per)

def del_existing(path_test, path_train):
    if os.path.exists(path_test) and len(os.listdir(path_test)) > 0:
        print("[+] Deleting existing test data")
        for f in os.listdir(path_test):
            os.remove(path_test+f)
        os.rmdir(path_test)
    if os.path.exists(path_train) and len(os.listdir(path_train)) > 0:
        print("[+] Deleting existing train data")
        for f in os.listdir(path_train):
            os.remove(path_train+f)
        os.rmdir(path_train)
# del_existing()
    
face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

def _trim_css_to_bounds(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

def _rect_to_css(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()

def get_image_count(path_test, path_train):
    return len(os.listdir(path_test))+len(os.listdir(path_train))

def create_faces(name):
    path_test = './faces/test/{}/'.format(name)
    path_train = './faces/train/{}/'.format(name)
    if not os.path.exists(path_test):
        os.mkdir(path_test)

    if not os.path.exists(path_train):
        os.mkdir(path_train)
    i=0
    while True:
        suc, frame = cap.read()
        # time.sleep(1/30)
        faces = []
        if suc:
            for face in face_detector(frame, 1):
                faces.append(_trim_css_to_bounds(_rect_to_css(face.rect), frame.shape))
            
            # print(faces)
            if len(faces) != 1:
                print("Skipping",i)            
                continue
            
            if get_image_count(path_test, path_train) >= 100:
                break

            if len(faces) == 1:
                for (y1,x2,y2,x1) in faces:
                    x1, y1, x2, y2 = x1-10, y1-10, x2+10, y2+10
                    roi = frame[y1:y2, x1:x2]
                    if roi.shape[0] > 150 and roi.shape[1] > 150:
                        roi = cv2.resize(roi,(224,224))
                        print(roi.shape)
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
                        
                        if (i+1)%test_per == 0:
                            print("test")
                            cv2.imwrite(path_test+'{}{}.jpg'.format(name,i),roi)
                        else:
                            cv2.imwrite(path_train+'{}{}.jpg'.format(name,i),roi)
            cv2.imshow("{}".format(name),frame)            
            i+=1
            print(i)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    name = sys.argv[1]
    path_test = './faces/test/{}/'.format(name)
    path_train = './faces/train/{}/'.format(name)
    del_existing(path_test, path_train)
    create_faces(name)
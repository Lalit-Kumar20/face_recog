
import cv2

import os

import numpy as np

subjects = ["", "lalit","unknown"]

def detect_face(img):
   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
   
    if (len(faces) == 0):
        return None, None
    
    (x, y, w, h) = faces[0]
    cropp = gray[y: y + h, x: x + w]
    dimmmm = (111, 111)
    rrrr = cv2.resize(cropp, dimmmm, interpolation=cv2.INTER_AREA)
    #
    #cv2.imshow("img",rrrr)
    #cv2.waitKey(1000)
    return rrrr, faces[0]


def detectface(img):
    gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    path = "D:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml"
    f = cv2.CascadeClassifier(path)

    fac = f.detectMultiScale(gr, scaleFactor=1.2, minNeighbors=5);

    if (len(faces) == 0):
        return None, None
    cv2.imshow('img',img)
    cv2.waitKey(1000)
    (x, y, w, h) = fac[0]
    cp = gr[y: y + h, x: x + w]
    di = (111, 111)
    rr = cv2.resize(cp, di, interpolation=cv2.INTER_AREA)
    #
    cv2.imshow('img', rr)
    cv2.waitKey(1000)
    return rr, fac[0]
def prepare_training_data(data_folder_path):

    dirs = os.listdir(data_folder_path)
    

    faces = []

    labels = []

    for dir_name in dirs:

        if not dir_name.startswith("s"):
            continue;
            
       
        label = int(dir_name.replace("s", ""))
        
        
        subject_dir_path = data_folder_path + "/" + dir_name
        
        
        subject_images_names = os.listdir(subject_dir_path)
        
      
        for image_name in subject_images_names:
            
           
            if image_name.startswith("."):
                continue;
            
            
            image_path = subject_dir_path + "/" + image_name

           
            image = cv2.imread(image_path)
            
           

            cv2.waitKey(100)
            
         
            face, rect = detect_face(image)
            
           
            if face is not None:
                
                faces.append(face)
            
                labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels



print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")


print("Total faces: ", len(faces))
print("Total labels: ", len(labels))



face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def predict(img):
   

    #cv2.imshow("img",img)

    face, rect = detectface(img)
    print(rect)
    
    label, confidence = face_recognizer.predict(face)
   
    label_text = subjects[label]
    print(confidence)
    if(confidence<120):
         print(label_text)
    else:
        print('unknown')
  
    return img



print("Predicting images...")


test_img1 = cv2.imread("C:\\Users\\LALIT\\Desktop\\opencv-face-recognition-python-master\\test-data\\test4.jpg")


predicted_img1 = predict(test_img1)
#predicted_img2 = predict(test_img2)
print("Prediction complete")

#cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
#cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()






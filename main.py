#from facereco import *
#from webdriver import *
import os
import cv2
import numpy as np
from selenium import webdriver
from time import sleep
import time

def web(username,password):
        driver = webdriver.Chrome('C:\\Users\\LALIT\\Desktop\\chromedriver.exe')
        driver.get("https://instagram.com")
        sleep(2)
        driver.find_element_by_xpath("//input[@name=\"username\"]").send_keys(username)
        sleep(2)
        driver.find_element_by_xpath("//input[@name=\"password\"]").send_keys(password)
        sleep(2)
        driver.find_element_by_xpath('//*[@id="react-root"]/section/main/article/div[2]/div[1]/div/form/div[4]/button/div').click()
        sleep(20000)
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
    # cv2.imshow("img",rrrr)
    # cv2.waitKey(1000)
    return rrrr, faces[0]


def detectface(new):
    path = "D:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml"
    faqtt = cv2.CascadeClassifier(path)
    print(new)

    img = cv2.imread(new)

    gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    featt = faqtt.detectMultiScale(gr, scaleFactor=1.2, minNeighbors=5);

    if (len(faces) == 0):
        return None, None
    #cv2.imshow('img', img)
    #cv2.waitKey(1000)

    (x, y, w, h) = featt[0]
    cp = gr[y: y + h, x: x + w]
    di = (111, 111)
    rr = cv2.resize(cp, di, interpolation=cv2.INTER_AREA)
    #
   # cv2.imshow('img', rr)
    cv2.waitKey(1000)
    return rr, featt[0]


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

    return faces, labels


print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))


def predict(img):
    # cv2.imshow("img",img)

    face, rect = detectface(img)
    #face = img
    print(rect)

    label, confidence = face_recognizer.predict(face)

    label_text = subjects[label]
    print(confidence)
    if (confidence < 100):
        ans = "lalit"
        print("YES")


    else:
        ans = "unknown"
        print("NO")

    return ans


print("Predicting images...")

# test_img1 = cv2.imread("C:\\Users\\LALIT\\Desktop\\opencv-face-recognition-python-master\\test-data\\test1.jpg")

# predicted_img1 = predict(test_img1)
#predicted_img2 = predict(test_img2)
print("Prediction complete")

# cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
# cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))



path2 = "D:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml"
fa = cv2.CascadeClassifier(path2)
scale = 1.2
min = 3
size = (50, 50)
font = cv2.FONT_HERSHEY_SIMPLEX
start_time = time.time()
duration = 10
cap = cv2.VideoCapture(0);
while (True):

    ret, frame = cap.read()
    cv2.imwrite("full.jpg",frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = fa.detectMultiScale(gray, scaleFactor=scale, minNeighbors=min)
    if (len(rects) == 1):
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (0, 0, 0), -1)
            cv2.putText(frame,
                        'face',
                        (x, y),
                        font, 2,
                        (0, 0, 255),
                        2,
                        1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imshow('CHECK', frame)
            crop_img = gray[y: y + h, x: x + w]
            dimm = (111, 111)
            rr = cv2.resize(crop_img, dimm, interpolation=cv2.INTER_AREA)
            cv2.imwrite("face.jpg", rr)
        test = "C:\\Users\\LALIT\\Desktop\\opencv-face-recognition-python-master\\face.jpg"

        if(int(time.time() - start_time) > duration):
               final =  predict(test)
               if (final == "lalit"):
                    print("Found")
                    flag=1
                    break
               else:
                    flag=0
                    print("NOT found")
                    break
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

if(flag==1):
    web("username","password")
else:
    img2 = cv2.imread('C:\\Users\\LALIT\\Desktop\\opencv-face-recognition-python-master\\failure.jpg')

    while(True):
        cv2.imshow("failure",img2)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

cv2.destroyAllWindows()
cap.release()


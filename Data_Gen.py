import cv2
from pathlib import Path
# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    if faces is ():
        return None
    # Crop all faces found
    for (x, y, w, h) in faces:
        x = x - 10
        y = y - 10
        cropped_face = img[y:y + h + 50, x:x + w + 50]
    return cropped_face
def saveImage(image, userName, userId, imgId):
    # Create a folder with the name as userName
    Path("./Dataset/{}".format(userName)).mkdir(parents=True, exist_ok=True)
    # Save the images inside the previously created folder
    cv2.imwrite(r"C:\Users\Aniket Verma\Desktop\Accenture\Deep-Learning-Face-Recognition-master\Dataset\{}\{}_{}.jpg".format(userName, userId, imgId), image)
    print(imgId)
    print("[INFO] Image : {} has been saved in folder : {}".format(
        imgId, userName))


# Initialize Webcam
cap = cv2.VideoCapture(0)
print("Enter the id and name of the person: ")
userId = input()
userName = input()
address="http://192.168.43.1:8080/video"
cap.open(address)
count = 1
# Collect 100 samples of your face from webcam input
while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        face = cv2.resize(face_extractor(frame), (400, 400))
        # face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # Save file in specified directory with unique name
        #file_name_path = r"C:\Users\Aniket Verma\Desktop\Accenture\Deep-Learning-Face-Recognition-master\Images"+ str(count) + '.jpg'
        #cv2.imwrite(file_name_path, face)
        saveImage(face, userName, userId, count)
        count += 1
       # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)
    else:
        print("Face not found")
        pass
    if cv2.waitKey(1) == 13 or count == 100:  # 13 is the Enter Key
        break
cap.release()
cv2.destroyAllWindows()
print("[INFO] Dataset has been created for {}".format(userName))
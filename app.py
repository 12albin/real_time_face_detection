import cv2
import streamlit as st
import numpy as np
from PIL import Image


def detect_faces_in_image(uploaded_image):
    Image_array=np.array(Image.open(uploaded_image))
    #pretrained xml file
    faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    #converting image to grayscale
    gray=cv2.cvtColor(Image_array,cv2.COLOR_BGR2GRAY)

    faces=faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5,minSize=(30,30))

    #drawing rectangle around detected faces
    for(x,y,w,h) in faces:
        cv2.rectangle(Image_array,(x,y),(x+w,y+h),(0,255,0),2) #2 is rectangle scale

    #display image 
    st.image(Image_array,channels="BGR",use_column_width=True)

def detect_face():
        faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
        cap=cv2.VideoCapture(0)
        while True:
             ret,frame=cap.read()#ret is boolean value
            
             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
             faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5,  minSize=(30, 30) )
             for(x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) #2 is rectangle scale
             cv2.imshow('Face Detection', frame)


             if cv2.waitKey(1) & 0xFF==ord("q"):
                  break
        cap.release()
        cv2.destroyAllWindows()
                  

             




st.title("face detection")
st.subheader("either open camera and Detect faces or Upload an image and detect faces")
if st.button("open camera"):
    detect_face()

uploaded_image=st.file_uploader("upload an image ",type=['jpg','jpeg','png'])
if uploaded_image is not None:
    detect_faces_in_image(uploaded_image)
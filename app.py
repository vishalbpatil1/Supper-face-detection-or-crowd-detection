'''run on Anaconda prompt streamlit run app.py'''
import streamlit as st
import pandas as pd
import numpy  as np
import cv2
import os
from keras.preprocessing import image
import tensorflow as tf
font = cv2.FONT_HERSHEY_PLAIN
# file path change according to your file location

img_save_path=os.getcwd()+'\\test_1.jpg'
img_output_path=os.getcwd()+'\\output.jpg'
weight_path=os.getcwd()+'\\yolov3_face_training_3000.weights'
cfg_path=os.getcwd()+'\\yolov3_face_testing.cfg'

# Load Yolo
# heading
st.markdown(f'''<center><h1 style="font-family:cursive;color:rgb(0,250,200);text-decoration-line:overline underline;text-decoration-style:double ;">AIS Solutions Pvt Ltd.</h1></center>''',unsafe_allow_html=True)
st.markdown(f'''<center><h1 style="background-color:DodgerBlue ;">Human Counting Using Opencv</h1><center>''',unsafe_allow_html=True)
# upload file where you have to test 

uploaded_file = st.file_uploader("Choose a person image  jpg file where you have to test", type="jpg")
if uploaded_file is not None:
    image_person_resize=image.load_img(uploaded_file,target_size=(700,700))
    img_array = image.img_to_array(image_person_resize)
    image.save_img(img_save_path, img_array)
    st.image(image_person_resize)
if st.button('proceed'):




        # LOAD YOLOV3 MODEL
        net = cv2.dnn.readNet(weight_path,cfg_path)

        # Name custom object
        classes = ['f','b']
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
        # Loading image
        img=cv2.imread(img_save_path)
        img=cv2.resize(img,(700,700))
        #img = cv2.resize(img, None, fx=0.4, fy=0.4)

        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)
        # Insert here the path of your image
        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.6:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    h = int(detection[2] * width)
                    w = int(detection[3] * height*0.8)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        count=0
        for i in range(len(boxes)):
            
            if i in indexes:
                x, y, w, h = boxes[i]
                #label = str(classes[class_ids[i]])
                confidence =round(confidences[i],2)
                #############
                count+=1

                cv2.rectangle(img,(x,y), (x+h,y+w), (255,250,250), 2)
                #cv2.putText(img,str(count), (x,y-5), font, 2, (0,0,255), 2)
                #cv2.putText(img,str(float(confidence)*100)+'%', (x,y-30), font, 1, (0,0,255), 2)
        cv2.imwrite(img_output_path,img)
        image_output= image.load_img(img_output_path,target_size=(700,800))
        st.image(image_output)
        st.markdown(f'''<center><div class="card text-white bg-info mb-1" style="width: 25rem">
        <div class="card-body">
        <h1 class="card-title">Total faces</h1>
        <p class="card-text">{(count):,d}</p>
        </div>
        </div><center>''', unsafe_allow_html=True)

        if len(indexes)==0:
            st.header('Face is not recognized')

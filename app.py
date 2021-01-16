from flask import Flask,render_template,request,Markup
import os
import pandas as pd
import numpy as np

app = Flask(__name__)

def main_function():
    import cv2
    import os
    from keras.models import load_model
    import numpy as np
    from pygame import mixer
    import time


    mixer.init()
    sound = mixer.Sound('alarm.wav')

    face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
    leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
    reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')



    lbl=['Close','Open']

    model = load_model('models/cnncat2.h5')
    path = os.getcwd()
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    count=0
    score=0
    thicc=2
    rpred=[99]
    lpred=[99]

    while(True):
        ret, frame = cap.read()
        height,width = frame.shape[:2] 

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
        left_eye = leye.detectMultiScale(gray)
        right_eye =  reye.detectMultiScale(gray)

        cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

        for (x,y,w,h) in right_eye:
            r_eye=frame[y:y+h,x:x+w]
            count=count+1
            r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye,(24,24))
            r_eye= r_eye/255
            r_eye=  r_eye.reshape(24,24,-1)
            r_eye = np.expand_dims(r_eye,axis=0)
            rpred = model.predict_classes(r_eye)
            if(rpred[0]==1):
                lbl='Open' 
            if(rpred[0]==0):
                lbl='Closed'
            break

        for (x,y,w,h) in left_eye:
            l_eye=frame[y:y+h,x:x+w]
            count=count+1
            l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
            l_eye = cv2.resize(l_eye,(24,24))
            l_eye= l_eye/255
            l_eye=l_eye.reshape(24,24,-1)
            l_eye = np.expand_dims(l_eye,axis=0)
            lpred = model.predict_classes(l_eye)
            if(lpred[0]==1):
                lbl='Open'   
            if(lpred[0]==0):
                lbl='Closed'
            break

        if(rpred[0]==0 and lpred[0]==0):
            score=score+1
            cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        
        else:
            score=score-1
            cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        
            
        if(score<0):
            score=0   
        cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        if(score>15):
            cv2.imwrite(os.path.join(path,'image.jpg'),frame)
            try:
                sound.play()
                
            except:
                pass
            if(thicc<16):
                thicc= thicc+2
            else:
                thicc=thicc-2
                if(thicc<2):
                    thicc=2
            cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    return "Service Running"


@app.route("/", methods = ['POST','GET'])
def home_():
    return render_template('index.html')

@app.route("/start", methods = ['POST','GET'])
def starting_page():
    res = None
    if request.method == "POST":
        res = main_function()
    return '''
    <html>
    <head>
    <title>started server</title>
    <link rel="stylesheet" href="/static/css/service_running.css"/>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    </head>
    <body>
    <center>
    <h1>{res}</h1>
    </center>
    <div class="flex recording">
   <div class="recording-circle recording-circle-100"></div>
      <div class="recording-circle recording-circle-120"></div>
      <div class="recording-circle recording-circle-150"></div>
      <div class="processing-circle processing-circle-140"></div>
      <div class="processing-circle processing-circle-160"></div>
    </div>
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    </body>
    </html>
    '''.format(res=res)


if __name__ == "__main__":
    app.run(debug=True)
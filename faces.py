import numpy as np
import cv2
import pickle
import requests
import time
def start_app():
    face_cascade = cv2.CascadeClassifier( 'cascades/data/haarcascade_frontalface_default.xml' )
    #eye_cascade = cv2.CascadeClassifier( 'cascades/data/haarcascade_eye.xml' )
    #smile_cascade = cv2.CascadeClassifier( 'cascades/data/haarcascade_smile.xml' )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read( "./recognizers/face-trainner.yml" )
    pesan = 'Ada seseorang tidak dikenal, silahkan lihat gambar'
    last_created_img = 0
    img_creation_interval = 5 * 1000


    labels = {"person_name": 10}
    with open( "pickles/face-labels.pickle", 'rb' ) as f:
        og_labels = pickle.load( f )
        labels = {v: k for k, v in og_labels.items()}

    cap = cv2.VideoCapture( 0 )
    img_counter = 0

    while (True):
        # Capture frame-by-frame
        k = cv2.waitKey( 1 )
        ret, frame = cap.read()
        gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
        faces = face_cascade.detectMultiScale( gray, scaleFactor=1.5, minNeighbors=5 )
        for (x, y, w, h) in faces:
            # print(x,y,w,h)
            roi_gray = gray[y:y + h, x:x + w]  # (ycord_start, ycord_end)
            roi_color = frame[y:y + h, x:x + w]

            # recognize? deep learned model predict keras tensorflow pytorch scikit learn
            id_, conf = recognizer.predict( roi_gray )
            current_mills = int(round(time.time()*1000))
            name=""
            if conf >= 4 and conf <= 85:
                # print(5: #id_)
                # print(labels[id_])
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                print("detected name " +name)
                color = (255, 255, 255)
                stroke = 2
                cv2.putText( frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA )

            if (name == None or name == ''):
                name = 'tidak-dikenal'
            if name == 'tidak-dikenal' and last_created_img+img_creation_interval<current_mills:
                img_name = "tidakdikenal/unknown_{}.png".format(current_mills)
                cv2.imwrite( img_name, frame )
                last_created_img = current_mills
                print( "tidak-dikenal detected, {} written!".format( img_name ) )
                img_counter += 1
                base_url = 'https://api.telegram.org/bot1548284382:AAHwrtqzcm7VdFZF_YhhG85KUgFJkSue41U/sendMessage?chat_id=-473733036&text="{}"'.format(
                    pesan )
                #requests.get( base_url )


                files = {'photo': open( img_name,'rb' )}

                res = requests.post(
                    'https://api.telegram.org/bot1548284382:AAHwrtqzcm7VdFZF_YhhG85KUgFJkSue41U/sendPhoto?chat_id=-473733036&caption= '
                    'Ada seseorang yang tidak dikenal, silahkan lihat gambar', files=files )
                print( res.status_code )


            #img_item = "5.jpg"
            #cv2.imwrite( img_item, roi_color )

            color = (255, 0, 0)  # BGR 0-255
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle( frame, (x, y), (end_cord_x, end_cord_y), color, stroke )
        # subitems = smile_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in subitems:
        #	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        # Display the resulting frame
        cv2.imshow( 'frame', frame )

        if cv2.waitKey( 20 ) & 0xFF == ord( 'q' ):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
def main():
    start_app()

if __name__ == "__main__":
    main()
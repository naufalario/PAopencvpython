import cv2
import keyboard
import os
from faces_train import train_dataset
from faces import start_app


def open_camera(user_dir):
    face_cascade = cv2.CascadeClassifier( 'cascades/data/haarcascade_frontalface_default.xml' )

    cap = cv2.VideoCapture( 0 )


    is_creating_file = False

    creating_file_delay = 0
    is_face_detected = False
    image_count = 0

    while (True):
        if is_creating_file:
            if creating_file_delay < 50:
                creating_file_delay += 1
            else:
                is_creating_file = False
                creating_file_delay = 0

        ret, frame = cap.read()
        frame_raw = frame.copy()
        gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
        faces = face_cascade.detectMultiScale( gray, scaleFactor=1.5, minNeighbors=5 )

        is_face_detected = False
        for (x, y, w, h) in faces:
            # print(x,y,w,h)
            color = (255, 0, 0)  # BGR 0-255
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle( frame, (x, y), (end_cord_x, end_cord_y), color, stroke )
            is_face_detected = True

        try:  # used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed( 'space' ):  # if key 'q' is pressed
                if not is_creating_file:
                    if is_face_detected:
                        image_count+=1
                        is_creating_file = True
                        print( 'Membuat dataset di: ' + user_dir  )
                        img_name = user_dir + "/{}.png".format(image_count)
                        cv2.imwrite( img_name, frame_raw )

                        if image_count == 5:
                            print('Dataset selesai dibuat!')
                            # cap.release()
                            # cv2.destroyAllWindows()
                            break


                    else:
                        print( "Wajah tidak terdeteksi" )
                else:
                    print( "Tunggu Sebentar..." )


        except:
            break  # if user pressed a key other than the given key the loop will break

        cv2.imshow( 'frame', frame )
        if cv2.waitKey( 20 ) & 0xFF == ord( 'q' ):
            break
    cap.release()
    cv2.destroyAllWindows()

def dataset_confirmation():
    print("Apakah ingin menambah orang lagi? (y/n)")
    response = None
    while response not in {"y", "n"}:
        response = input("Masukkan y atau n: ")
    if (response == "y"):
        username = input( "Masukkan nama:" )
        BASE_DIR = os.path.dirname( os.path.abspath( __file__ ) )
        image_dir = os.path.join( BASE_DIR, username )
        user_dir ='dataset/' + username
        if not os.path.exists( os.path.join( username, image_dir ) ):
            os.makedirs( user_dir )
        open_camera(user_dir)
        dataset_confirmation()
    else:
        train_dataset()
        print("Selesai memproses dataset")
        start_app()

username = input( "Masukkan nama:" )
print( "Namamu adalah: " + username )
BASE_DIR = os.path.dirname( os.path.abspath( __file__ ) )
image_dir = os.path.join( BASE_DIR, username )
user_dir ='dataset/' + username
if not os.path.exists( os.path.join( username, image_dir ) ):
    os.makedirs( user_dir )
open_camera(user_dir)
dataset_confirmation()



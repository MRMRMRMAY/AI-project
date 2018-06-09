# Proof-of-concept
import cv2
import sys
from constants import *
from emotion_recognition import EmotionRecognition
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import threading
import musicPlayer
import time



cascade_classifier = cv2.CascadeClassifier(CASC_PATH)

def brighten(data, b):
    datab = data * b
    return datab

def format_image(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=5
    )
    # None is we don't found an image
    if not len(faces) > 0:
        return None
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    # Chop image to face
    face = max_area_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
    # Resize image to network size
    try:
        image = cv2.resize(image, (SIZE_FACE, SIZE_FACE),
                           interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("[+] Problem during resize")
        return None
    # cv2.imshow("Lol", image)
    # cv2.waitKey(0)
    return image

class SmartBudy(tk.Tk):
    def initEmotionObverserLabel(self):
        self.emotion_label_list = []
        for e in EMOTIONS:
            label = tk.Label(self.label_list_grid, text = e + ":", justify=tk.LEFT)
            label.grid(row = EMOTIONS.index(e), column = 0)
            self.emotion_label_list.append(label)
    def __init__(self):
        super(SmartBudy,self).__init__()
        self.center_window(self,width=1024, height=600)
        self.title("SMART BUDY")
        self.grid_view = tk.Frame(self, width = 1024, height = 600)
        self.grid_view.grid(row = 0, column = 0)
        self.usageLabel = tk.Label(self.grid_view, text='{0}{1}{2}{3}'.format("z: turn on camera\n",
                                                                    "x: turn off camera\n",
                                                                    # "c: pause/resume the music\n",
                                                                    "v: stop the music\n",
                                                                    "q: quit the system\n"), justify=tk.LEFT)
        self.usageLabel.grid(row=0, column=0)
        self.camera_view_canvas = tk.Canvas(self.grid_view, width=700, height=600)
        self.camera_view_canvas.grid(row=0, column=1)
        self.label_list_grid = tk.Frame(self, width = 50, height = 600,borderwidth = 20)
        self.label_list_grid.grid(row = 0, column = 2)
        self.initEmotionObverserLabel()
        self.bind('<KeyRelease>', self.onkey_callback)
    def center_window(self,root, width, height):
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        size = '%dx%d+%d+%d' % (width, height, (screen_width - width) / 2, (screen_height - height) / 2)
        root.geometry(size)
    def onkey_callback(self,event):
        global camera_lock
        if event.char == 'q':
            print('[+]quit')
            sys.exit()
        elif event.char == 'z' and camera_lock.acquire():
            emotion_reco_thread = threading.Thread(target= self.emotion_reco())
            emotion_reco_thread.start()
            emotion_reco_thread.join()
            print('[+]turn on the camera and emotion recognition model')
        elif event.char == 'x':
            if camera_lock.locked():
                camera_lock.release()
            for i in range(len(self.emotion_label_list)):
                self.emotion_label_list[i].config(text = EMOTIONS[i]+" :")
            print('[+]turn off the camera and emotion recognition model')
        elif event.char == 'c':
            print('[+]c')
        elif event.char == 'v':
            print('[+]stop the music')
            musicplayer.stop()
    def emotion_reco(self): # emotion recognition process and camera capture process
        global camera_lock,musicplayer
        video_capture = cv2.VideoCapture(0)
        sad_count = 0
        happy_count = 0
        while camera_lock.locked():
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            # Predict result with network
            result = network.predict(format_image(frame))
            # Write results in frame
            if result is not None:
                for i in range(len(result[0])):
                    self.emotion_label_list[i].config(text = EMOTIONS[i] + " : " + str(result[0][i]))
                for index, emotion in enumerate(EMOTIONS):
                    cv2.putText(frame, emotion, (10, index * 20 + 20),
                                cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
                    cv2.rectangle(frame, (130, index * 20 + 10), (130 +
                                                                  int(result[0][index] * 100), (index + 1) * 20 + 4), (255, 0, 0), -1)
                face_image = feelings_faces[np.argmax(result[0])]
               
                # Ugly transparent fix
              
                for c in range(0, 3):
                    frame[200:320, 10:130, c] = face_image[:, :, c] * \
                        (face_image[:, :, 3] / 255.0) + frame[200:320,
                                                              10:130, c] * (1.0 - face_image[:, :, 3] / 255.0)
                                                              
                '''to give a response depended on the emotion'''
                if np.argmax(result[0]) == SAD: #detect the sad emotion 50 times
                    sad_count += 1
                    if sad_count >= 3:
                        happy_count = 0
                    if not musicplayer.isPlaying(): # music player is empty
                        if sad_count >= 10:
                            musicplayer.load(HAPPY)
                            musicplayer.run()
                            print("music player")
                            sad_count = 0
                    else: #playing the music
                        sad_count = 0
                if np.argmax(result[0]) == HAPPY:
                    happy_count += 1
                    if musicplayer.emotionMark == HAPPY and happy_count >= 50:# detect the happy emotion 50 times
                        musicplayer.stop()
                        happy_count = 0
                        sad_count = 0
                print("result: ", EMOTIONS[np.argmax(result[0])])
            '''display in the tk'''
            camera_view = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_view = Image.fromarray(camera_view)
            camera_view = ImageTk.PhotoImage(image= camera_view)
            self.camera_view_canvas.create_image(camera_view.width()/2+100, camera_view.height()/2, image = camera_view)
            self.update_idletasks()
            self.update()
        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()
if __name__ == '__main__':
    # Load Model
    network = EmotionRecognition()
    network.build_network()# build AlexNet, a kind of CNN

    musicplayer = musicPlayer.musicPlayer()
    camera_lock = threading.Lock()

    font = cv2.FONT_HERSHEY_SIMPLEX
    #load the emotion icon

    feelings_faces = []
    for index, emotion in enumerate(EMOTIONS):
        feelings_faces.append(cv2.imread('./emojis/' + emotion + '.png', -1))

    obj = SmartBudy()
    obj.mainloop()


    #GUI part
    # root = tk.Tk()
    # gui.center_window(root, width=1024, height = 600)
    # root.title("SMART BUDY") # set window title
    # grid_view = tk.Canvas(root, width = 1024, height = 600)
    # grid_view.grid(row = 0, column = 0)
    # usageLabel = tk.Label(grid_view, text='{0}{1}{2}{3}'.format("z: turn off camera\n",
    #                                                            "x: turn on camera\n",
    #                                                            # "c: pause/resume the music\n",
    #                                                            "v: stop the music\n",
    #                                                            "q: quit the system\n"), justify='left')
    # usageLabel.grid(row=0, column=0)
    # camera_view_canvas = tk.Canvas(grid_view, width = 900, height = 600)
    # camera_view_canvas.grid(row = 0, column = 1)
    # def onkey_callback(event):
    #     global camera_lock
    #     if event.char == 'q':
    #         print('[+]quit')
    #         sys.exit()
    #     elif event.char == 'z' and camera_lock.acquire():
    #         emotion_reco_thread = threading.Thread(target=emotion_reco())
    #         emotion_reco_thread.start()
    #         emotion_reco_thread.join()
    #         print('[+]turn on the camera and emotion recognition model')
    #     elif event.char == 'x':
    #         if camera_lock.locked():
    #             camera_lock.release()
    #         print('[+]turn off the camera and emotion recognition model')
    #     elif event.char == 'c':
    #         print('[+]c')
    #     elif event.char == 'v':
    #         print('[+]stop the music')
    #         musicplayer.stop()
    # root.bind('<KeyRelease>',onkey_callback)
    # def emotion_reco():
    #     global camera_lock,root,camera_view_canvas,musicplayer
    #     video_capture = cv2.VideoCapture(0)
    #     sad_count = 0
    #     happy_count = 0
    #     while camera_lock.locked():
    #         # Capture frame-by-frame
    #         ret, frame = video_capture.read()
    #         # Predict result with network
    #         result = network.predict(format_image(frame))
    #         # Write results in frame
    #         if result is not None:
    #             for index, emotion in enumerate(EMOTIONS):
    #                 cv2.putText(frame, emotion, (10, index * 20 + 20),
    #                             cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
    #                 cv2.rectangle(frame, (130, index * 20 + 10), (130 +
    #                                                               int(result[0][index] * 100), (index + 1) * 20 + 4), (255, 0, 0), -1)
    #             face_image = feelings_faces[np.argmax(result[0])]
    #             # Ugly transparent fix
    #             for c in range(0, 3):
    #                 frame[200:320, 10:130, c] = face_image[:, :, c] * \
    #                     (face_image[:, :, 3] / 255.0) + frame[200:320,
    #                                                           10:130, c] * (1.0 - face_image[:, :, 3] / 255.0)
    #             '''to give a response depended on the emotion'''
    #             if np.argmax(result[0]) == SAD: #detect the sad emotion 50 times
    #                 sad_count += 1
    #                 happy_count = 0
    #                 if not musicplayer.isPlaying(): # music player is empty
    #                     if sad_count >= 10:
    #                         musicplayer.load(HAPPY)
    #                         musicplayer.run()
    #                         print("music player")
    #                         sad_count = 0
    #                 else: #playing the music
    #                     sad_count = 0
    #             if np.argmax(result[0]) == HAPPY:
    #                 happy_count += 1
    #                 if musicplayer.emotionMark == HAPPY and happy_count >= 50:# detect the happy emotion 50 times
    #                     musicplayer.stop()
    #                     happy_count = 0
    #                     sad_count = 0
    #             print("result: ", EMOTIONS[np.argmax(result[0])])
    #         '''display in the tk'''
    #         camera_view = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         camera_view = Image.fromarray(camera_view)
    #         camera_view = ImageTk.PhotoImage(image= camera_view)
    #         camera_view_canvas.create_image(camera_view.width()/2+100, camera_view.height()/2, image = camera_view)
    #         root.update_idletasks()
    #         root.update()
    #     # When everything is done, release the capture
    #     video_capture.release()
    #     cv2.destroyAllWindows()
    # root.mainloop()




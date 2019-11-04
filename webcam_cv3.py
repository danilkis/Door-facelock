import cv2,sys,logging as log,datetime as dt 
from time import sleep

cascPath = "lbpcascade_frontalface_improved.xml"                    #Путь к каскаду
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)
def face_find():
    video_capture = cv2.VideoCapture(0)                             #Начало захвата видео
    anterior = 0                                                    #Кол-во лиц
    while anterior == 0 :
        if not video_capture.isOpened():                            #При оцутствии подключении к камере выдавать ошибку
            print('Проверьте подключение к камере')
            sleep(5)
            pass
    
        ret, frame = video_capture.read()                           #Захват видео по кадрам

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:                                  #Рисуем рамку у лица
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 5, 5), 2)

        if anterior != len(faces):
            anterior = len(faces)
            log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

        cv2.imshow('Видеоключ', frame)                              #Показываем готовый результат

        if cv2.waitKey(1) & 0xFF == ord('q'):                       #Выход по желанию
            break

        if anterior >= 1:                                           #Если обнаружено одно или более лиц открываем дверь
            print(anterior," Лиц обнаружено")
            print("Открываю")

    video_capture.release()                                         #Когда цикл завершен завершаем все
    cv2.destroyAllWindows()                                         #Закрываем окно с видео
face_find()




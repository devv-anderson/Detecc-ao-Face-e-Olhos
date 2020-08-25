import numpy as np
import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #carregamento arquivo face
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') #carregamento arquivo olhos

while(True):
	_, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #frame bp
	faces = face_cascade.detectMultiScale(gray, 1.3, 5) #deteccao do rosto, do frame gray, 1.3 e 5 parametros padrão

	for (x,y,w,h) in faces: #inicio e final da face,  altura e largura
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #parametros de cor
		roi_gray = gray[y:y+h, x:x+w] #recorte do rosto pb
		roi_color = frame[y:y+h, x:x+w] #recorte do rosto corolido
		cv2.imshow('roi_gray', roi_gray) #mostrar mini recorte
		eyes = eye_cascade.detectMultiScale(roi_gray) #recorte dos olhos
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)



	cv2.imshow('frame',frame)

	key = cv2.waitKey(1)

	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()

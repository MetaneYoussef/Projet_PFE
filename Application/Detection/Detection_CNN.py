from cvzone.FaceDetectionModule import FaceDetector
import cv2
import tensorflow as tf
import numpy as np

    
cap = cv2.VideoCapture(0) #Flux de la webcam
detector = FaceDetector() 
classes = ['Masque Incorrect', 'Sans Masque', 'Masque Correct']

# importation du modele
new_model = tf.keras.models.load_model('./model_cnn.h5')

# initialisation de l'écriture sur l'image
writing = ['']


while True:

    #initialisation des variable pour chaque image (Frame)
    success, img = cap.read()
    predictions = np.array([])
    cropped_image = []
    crpd_img = np.array([])

    # detection du visage + dessin des rectangles sur l'image
    img, bboxs = detector.findFaces(img) 

    #S'il y a des visages dans l'image
    if len(bboxs)>0: 

        for i in range(len(bboxs)): # pour chaque visages

            box = bboxs[i]['bbox']

            # prenant l'image recadré du visage
            cr_img = img[   box[1]:box[1]+box[3]+10  ,  box[0]:box[0]+box[2] ]
                                    #height                         #width

            # Verifier Si tous le visage est à l'intérieur de l'image
            if cr_img is None or box[0]-50 < 0 or box[1]-50 < 0 or box[0]+box[2] > img.shape[1] or box[1]+box[3] > img.shape[0]:
                continue
            else: 
                
                cr_img = cv2.resize(cr_img, dsize=(224,224)) # redimensionnement de l'image du visage
                
                cropped_image.append(cr_img) #ajouter l'image à la liste des images

                # modifier les dimension de la liste pour la rendre compatible avec la fonction du prediction
                crpd_img = np.concatenate((cropped_image,),axis = 1) 
        
        # s'il  ya des images pour la prediction
        if (len(crpd_img)>0):

            predictions = new_model.predict(crpd_img) # faire la prediction
            predictions = tf.nn.sigmoid(predictions).numpy() #tranformation d'un objet tensor en une liste numpy
            
            #pour chaque prediction d'un visage
            for j in range(len(predictions)): 

                truth = classes[np.argmax(predictions[j])] #avoir la valeur de la classe
                
                # ne faire la màj sauf si le pourcentage est supérieur à 90 %
                if np.max(predictions[j])> 0.9 :
                    if j >= len(writing):
                        writing.append( truth +' '+str(np.max(predictions[j])*100)[:2]+'%')
                    else:
                        writing[j] = truth +' '+str(np.max(predictions[j])*100)[:2]+'%'

                
            #traitement des cas sp
            # éciales           
            if len(writing) > len(predictions):
                writing = writing[:len(predictions)]
            elif len(writing) < len(predictions):
                for i in range(len(predictions)):
                    if j >= len(writing):
                        writing.append( truth +' '+str(np.max(predictions[j])*100)[:2]+'%')
                    else:
                        writing[j] = truth +' '+str(np.max(predictions[j])*100)[:2]+'%'

            # écrire la prediction sur l'image
            for i in range(len(writing)):
                cv2.putText(img,writing[i],(bboxs[i]['bbox'][0],bboxs[i]['bbox'][1]),cv2.FONT_HERSHEY_DUPLEX,1,(0,150,0),2)
           
    cv2.imshow("Detection Avec Modele", img) # afficher l'image (Frame)
    
   
    if cv2.waitKey(1) & 0xFF == ord('q'): #attendre la touche q pour 
        break
cap.release()
cv2.destroyAllWindows()

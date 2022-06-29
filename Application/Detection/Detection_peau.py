from cv2 import imshow
from cvzone.FaceDetectionModule import FaceDetector
import cv2
import numpy as np


def skin_mask(cr_img,nose = False,mouth = False):
    min_YCrCb = np.array([0,135,58],np.uint8)
    max_YCrCb = np.array([255,180,135],np.uint8)
    min_HSV = np.array([0, 15, 0], dtype = "uint8")
    max_HSV = np.array([17, 150, 255], dtype = "uint8")
    if nose:
        crop = cr_img[  int(cr_img.shape[0]*(1/3)):int(cr_img.shape[0]*(2/3))-4  ,   20:int(cr_img.shape[1])-20]
    if mouth:
        crop = cr_img[  int(cr_img.shape[0]*(2/3)):int(cr_img.shape[0])-4  ,   20:int(cr_img.shape[1])-20]
    else:
        crop = cr_img[  int(cr_img.shape[0]*(1/3)):int(cr_img.shape[0])-4  ,   20:int(cr_img.shape[1])-20]
    total_area = int(crop.shape[0])*int(crop.shape[1])
    imageHSV = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    imageYCrCb = cv2.cvtColor(crop,cv2.COLOR_BGR2YCR_CB)
    Ycrcb_mask = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
    hsv_mask = cv2.inRange(imageHSV, min_HSV, max_HSV)
    mask = cv2.bitwise_or(Ycrcb_mask,hsv_mask)
    mask = cv2.medianBlur(mask,3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))
    count=np.count_nonzero(mask)

    return mask,total_area,count


cap = cv2.VideoCapture(0)
detector = FaceDetector()



while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img)
    cropped_image = []
    Hori = np.array([])
    if len(bboxs)>0:
        print(bboxs)
        for i in range(len(bboxs)):
            box = bboxs[i]['bbox']
            cr_img = img[   box[1]:box[1]+box[3]    ,  box[0]:box[0]+box[2] ]
                                    #height                         #width

            if cr_img is None or box[0]-50 < 0 or box[1]-50 < 0 or box[0]+box[2] > img.shape[1] or box[1]+box[3] > img.shape[0]:
                continue
            else:
                 
               
                mask,total_area,count = skin_mask(cr_img)
                cr_img = cv2.resize(cr_img, dsize=(224,224))
                
                area = int((count/total_area)*100)

                if count/total_area < 0.15:
                    cv2.putText(img,"Maks worn Correctly "+str(area)+" %",(bboxs[i]['bbox'][0],bboxs[i]['bbox'][1]),cv2.FONT_HERSHEY_DUPLEX,1,(0,150,0),2)
                elif count/total_area > 0.65:
                    cv2.putText(img,"No Mask "+str(area)+" %",(bboxs[i]['bbox'][0],bboxs[i]['bbox'][1]),cv2.FONT_HERSHEY_DUPLEX,1,(0,150,0),2)
                else :
                    mask,total_area,count = skin_mask(cr_img,nose=True)
                    
                    pourcentage = str(int((count/total_area)*100))
                    if count/total_area > 0.3:
                        cv2.putText(img,"Maks worn Incorrectly "+str(area)+" %",(bboxs[i]['bbox'][0],bboxs[i]['bbox'][1]),cv2.FONT_HERSHEY_DUPLEX,1,(0,150,0),2)
                    else : 
                        mask,total_area,count = skin_mask(cr_img,mouth=True)
                        imshow("mask",mask)
                        area = int((count/total_area)*100) 
                        if count/total_area > 0.4:
                            cv2.putText(img,"Maks worn Incorrectly "+str(area)+" %",(bboxs[i]['bbox'][0],bboxs[i]['bbox'][1]),cv2.FONT_HERSHEY_DUPLEX,1,(0,150,0),2)
                        else:
                            cv2.putText(img,"Maks worn Correctly "+str(area)+" %",(bboxs[i]['bbox'][0],bboxs[i]['bbox'][1]),cv2.FONT_HERSHEY_DUPLEX,1,(0,150,0),2)

                cropped_image.append(cr_img)
                Hori = np.concatenate(tuple(cropped_image), axis=1)

                cv2.imshow('HORIZONTAL', Hori)
        
    cv2.imshow("Image", img)
    
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
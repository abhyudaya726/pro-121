import cv2
import time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))
cap = cv2.VideoCapture(0)
time.sleep(2)
bg = 0

for i in range(60):
    ret,bg = cap.read()

print(ret,bg)

bg = np.flip(bg,axis=1)

while(cap.isOpened()):
    ret,img = cap.read()
    if not ret:
        break
    img = np.flip(img,axis=1)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower_red = np.array([10,120,50])
    upper_red = np.array([10,255,255])
    mask_1 = cv2.inRange(hsv,lower_red,upper_red)
    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask_2 = cv2.inRange(hsv,lower_red,upper_red)
    mask = mask_1 + mask_2
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask = cv2.morphologyEx(mask,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))
    mask_3 = cv2.bitwise_not(mask)
    image1 = cv2.bitwise_and(img,img,mask=mask_3)
    image2 = cv2.bitwise_and(bg,bg,mask=mask)
    final_output = cv2.addWeighted(image1,1,image2,1,0)
    output_file.write(final_output)
    cv2.imshow("invisibility",final_output)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
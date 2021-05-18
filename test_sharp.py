import cv2
import numpy as np

img=cv2.imread("rawframes/1/3.jpg")

# 1
blur=cv2.GaussianBlur(img,(0,0),3)
image=cv2.addWeighted(img,1.5,blur,-0.5,0)
# 2
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
image = cv2.filter2D(img, -1, kernel)
# 3
image=cv2.bilateralFilter(img,9,75,75)
# 4
sigma = 1; threshold = 5; amount = 1
blurred=cv2.GaussianBlur(img,(0,0),1,None,1)
lowContrastMask = abs(img - blurred) < threshold
sharpened = img*(1+amount) + blurred*(-amount)
image=cv2.bitwise_or(sharpened.astype(np.uint8),lowContrastMask.astype(np.uint8))

cv2.namedWindow("dst",cv2.WINDOW_FREERATIO)
cv2.imshow("dst",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
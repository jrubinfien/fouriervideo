import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist


movie = 'Julian_Rubinfien_ex2.mp4'

cap = cv2.VideoCapture(movie)
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


cnt = 0

out = cv2.VideoWriter(movie.split('.')[0]+'_fft.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 24, (frame_width,frame_height),0)


while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fft = np.fft.fftshift(np.fft.fft2(gray))
        spec_img_log = np.log(1+abs(fft))
        normalized = 255*(spec_img_log-np.min(spec_img_log))/(np.max(spec_img_log)-np.min(spec_img_log))


        normuint = np.uint8(normalized)

        out.write(normuint)

        cv2.imshow('frame',normuint)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()
out.release()
 
cv2.destroyAllWindows()
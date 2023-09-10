import cv2
import numpy as np

cap = cv2.VideoCapture('cap.mkv')


# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
record = 1

# Read until video is completed
frame_count = 1
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    if record:
        imgH = frame.shape[0]
        imgW = frame.shape[1]
        video_write=cv2.VideoWriter('video_opencv.avi',cv2.VideoWriter_fourcc('M','J','P','G'),60,(imgW, imgH))
        record = 0

    # Display the resulting frame
    # cv2.imshow('Frame',frame)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

    video_write.write(frame)
    
    if frame_count%1200 == 0:
        print(frame_count)
        break
    frame_count +=1

  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
video_write.release()

# Closes all the frames
cv2.destroyAllWindows()


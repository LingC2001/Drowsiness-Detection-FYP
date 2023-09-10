import cv2


cap = cv2.VideoCapture("video.avi")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print("initial fps: " + str(fps))
out_fps = 60
duration_scale = 2
filler_frames = int((out_fps//(fps//2)) - 1)
print("number of filler frame to get " + str(out_fps) + " fps:" + str(filler_frames))



out = cv2.VideoWriter('cap_20_night.mp4',cv2.VideoWriter_fourcc('m','p','4','v'), out_fps, (width,height))

if (cap.isOpened()== False): 
    print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        out.write(frame)
        for i in range(filler_frames):
            out.write(frame)
        
    # Break the loop
    else: 
        break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()
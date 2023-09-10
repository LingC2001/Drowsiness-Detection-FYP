# libraries
import cv2
import mediapipe as mp
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter,filtfilt, find_peaks
from tqdm import tqdm
import sys

# Constant
LEFT_EYE = [362, 385, 387, 263, 373, 380]    # p1, p2, p3, p4, p5, p6
RIGHT_EYE = [33, 160, 158, 133, 153, 144]    # p1, p2, p3, p4, p5, p6
EAR_THRESHOLD = 0.004   # Adjust this to get a good threshold


def draw_features(img, landmark_lst, width, height, util):
    '''
    Draws eye features onto images.
    img: OpenCV images
    landmark_lst: landmarks from Mediapipe solutions
    width: width of img
    height: height of img
    util: mp.solutions.drawing_utils
    '''
    if not landmark_lst:
        return

    # draw left eye features
    for idx in LEFT_EYE:
        landmark = landmark_lst.landmark[idx]
        img_coord = util._normalized_to_pixel_coordinates(landmark.x, landmark.y, width, height)
        cv2.circle(img, img_coord, 3, (255, 255, 255), -1)

    # draw right eye features
    for idx in RIGHT_EYE:
        landmark = landmark_lst.landmark[idx]
        img_coord = util._normalized_to_pixel_coordinates(landmark.x, landmark.y, width, height)
        cv2.circle(img, img_coord, 3, (255, 255, 255), -1)
    
    return


def distance(point1, point2):
    '''
    Calculates the euclidean distance betweeen point1 and point2
    point1: landmark from Mediapipe solution
    point2: landmark from Mediapipe solution
    return: euclidean distance
    '''
    x1, y1 = point1.x, point2.y
    x2, y2 = point2.x, point2.y
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


def calculate_ear(landmark_lst, verbose=False):
    '''
    Calculates the EAR based on landmarks
    landmark_lst: list of landmarks from Mediapipe solution
    verbose: prints 'Eye is closed' when EAR is below threshold, doesn't print by default
    return: EAR score for the current frame
    '''
    if not landmark_lst:
        return 0
    
    # left eye distances
    vertical1_left = distance(landmark_lst.landmark[LEFT_EYE[1]], landmark_lst.landmark[LEFT_EYE[5]])
    vertical2_left = distance(landmark_lst.landmark[LEFT_EYE[2]], landmark_lst.landmark[LEFT_EYE[4]])
    horizontal_left = distance(landmark_lst.landmark[LEFT_EYE[0]], landmark_lst.landmark[LEFT_EYE[3]])

    # right eye distances
    vertical1_right = distance(landmark_lst.landmark[RIGHT_EYE[1]], landmark_lst.landmark[RIGHT_EYE[5]])
    vertical2_right = distance(landmark_lst.landmark[RIGHT_EYE[2]], landmark_lst.landmark[RIGHT_EYE[4]])
    horizontal_right = distance(landmark_lst.landmark[RIGHT_EYE[0]], landmark_lst.landmark[RIGHT_EYE[3]])

    # calculate EAR score
    ear_left = 0.5*(vertical1_left + vertical2_left)/horizontal_left
    ear_right = 0.5*(vertical1_right + vertical2_right)/horizontal_right
    ear_score = 0.5*(ear_left + ear_right)

    if verbose and ear_score < EAR_THRESHOLD:
        print('Eye is closed')
    
    return ear_score

def butter_bandpass_filter(data, cutoff, fs, order):
    '''
    Butterworth bandpass filter
    data: data to be filtered
    cutoff: (has to be a numpy array) the two cutoff frequencies for the bandpass filter
    fs: sampling rate
    order: order of the filter
    return: filtered data
    '''
    normal_cutoff = cutoff / (fs*0.5)
    b, a = butter(order, normal_cutoff, btype='bandpass', analog=False)
    y = filtfilt(b, a, data)
    return y


def main():
    # Mediapipe functions
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    # open video file and get total frames + fps
    cap = cv2.VideoCapture(sys.argv[1])
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fs = cv2.CAP_PROP_FPS

    # initialise EAR array
    close_data = np.array([None]*total_frames)
    idx = 0

    # start face_mesh function
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        # tqdm for progress bar
        with tqdm(total=total_frames) as pbar:
            while cap.isOpened():
                # read image
                success, image = cap.read()
                if not success:
                    break

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)      # landmarks

                # Draw the face mesh annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]    # only face in list since max_num_faces=1
                    
                    # drawing eye on face
                    # draw_features(image, face_landmarks, image.shape[1], image.shape[0], mp_drawing)

                    # count blinking rate
                    score = calculate_ear(face_landmarks, verbose=False)
                    close_data[idx] = score
                else:
                    close_data[idx] = None
                idx += 1
                pbar.update(1)
    
                # cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
                # if cv2.waitKey(5) & 0xFF == 27:
                #     break
                
            cap.release()
    
    # pad when EAR is NaN
    close_data = pd.Series(close_data)
    close_data = close_data.fillna(method='pad', limit=None)
    close_data = np.array(close_data)

    # filter data
    filtered_data = butter_bandpass_filter(close_data, np.array([0.1, 1.2]), fs, 5)

    # find peaks
    peak_idx, _ = find_peaks(-filtered_data, height=EAR_THRESHOLD, distance=5, prominence=(None, None))
    print(f'Total blinks: {len(peak_idx)}')
    
    # plot data
    t_stamp = np.array([i/fs for i in range(len(filtered_data))])
    plt.plot(t_stamp, filtered_data, label='EAR')
    plt.plot(t_stamp[peak_idx], filtered_data[peak_idx], "x", label='Blink')
    plt.title(f'Total blinks: {len(peak_idx)}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Eye Aspect Ratio')
    plt.legend()
    plt.show()

    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
"""
Analyses all mp4 or mkv files in the specified folder
Example run:
python save_face_features.py --folder_path "folder/path"

This program extracts face data (landmarks, EAR, MAR, PUC, MOE) for all videos in a given folder and save them into 2 files
"face_landmarks.txt": Stores time and the 478 normalised face mesh landmark positions (x, y, z)
"face_features.txt": Stores some handcrafted/analyzed features such as EAR, MAR, PUC, MOE

"""


import csv
import cv2
import mediapipe as mp
import math
import numpy as np
from tqdm import tqdm 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt, find_peaks
import argparse
import os

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def distance(p1, p2):
    ''' Calculate distance between two points
    :param p1: First Point 
    :param p2: Second Point
    :return: Euclidean distance between the points. (Using only the x and y coordinates).
    '''
    return (((p1[:2] - p2[:2])**2).sum())**0.5

def eye_aspect_ratio(landmarks, eye):
    ''' Calculate the ratio of the eye length to eye width. 
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :param eye: List containing positions which correspond to the eye
    :return: Eye aspect ratio value
    '''
    N1 = distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
    N2 = distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
    N3 = distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
    D = distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
    return (N1 + N2 + N3) / (3 * D)

def eye_feature(landmarks):
    ''' Calculate the eye feature as the average of the eye aspect ratio for the two eyes
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :return: Eye feature value
    '''
    return (eye_aspect_ratio(landmarks, left_eye) + \
    eye_aspect_ratio(landmarks, right_eye))/2

def mouth_feature(landmarks):
    ''' Calculate mouth feature as the ratio of the mouth length to mouth width
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :return: Mouth feature value
    '''
    N1 = distance(landmarks[mouth[1][0]], landmarks[mouth[1][1]])
    N2 = distance(landmarks[mouth[2][0]], landmarks[mouth[2][1]])
    N3 = distance(landmarks[mouth[3][0]], landmarks[mouth[3][1]])
    D = distance(landmarks[mouth[0][0]], landmarks[mouth[0][1]])
    return (N1 + N2 + N3)/(3*D)

def pupil_circularity(landmarks, eye):
    ''' Calculate pupil circularity feature.
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :param eye: List containing positions which correspond to the eye
    :return: Pupil circularity for the eye coordinates
    '''
    perimeter = distance(landmarks[eye[0][0]], landmarks[eye[1][0]]) + \
                    distance(landmarks[eye[1][0]], landmarks[eye[2][0]]) + \
                    distance(landmarks[eye[2][0]], landmarks[eye[3][0]]) + \
                    distance(landmarks[eye[3][0]], landmarks[eye[0][1]]) + \
                    distance(landmarks[eye[0][1]], landmarks[eye[3][1]]) + \
                    distance(landmarks[eye[3][1]], landmarks[eye[2][1]]) + \
                    distance(landmarks[eye[2][1]], landmarks[eye[1][1]]) + \
                    distance(landmarks[eye[1][1]], landmarks[eye[0][0]])
    area = math.pi * ((distance(landmarks[eye[1][0]], landmarks[eye[3][1]]) * 0.5) ** 2)
    return (4*math.pi*area)/(perimeter**2)

def pupil_feature(landmarks):
    ''' Calculate the pupil feature as the average of the pupil circularity for the two eyes
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :return: Pupil feature value
    '''
    return (pupil_circularity(landmarks, left_eye) + \
            pupil_circularity(landmarks, right_eye))/2

def get_face_features():
    if results.multi_face_landmarks:
        
        ear = eye_feature(landmarks_positions)
        mar = mouth_feature(landmarks_positions)
        puc = pupil_feature(landmarks_positions)
        moe = mar/ear
        
    else:
        ear = -1000
        mar = -1000
        puc = -1000
        moe = -1000

    return ear, mar, puc, moe

def get_head_features():
    if results.multi_face_landmarks:
        SideL = landmarks_positions_raw[Side_Left[0]][2]
        SideR = landmarks_positions_raw[Side_Right[0]][2]
        HeadTopZ = landmarks_positions_raw[HeadTop[0]][2]
        HeadBotZ = landmarks_positions_raw[HeadBot[0]][2]
        HeadTopY = landmarks_positions_raw[HeadTop[0]][1]
        HeadBotY = landmarks_positions_raw[HeadBot[0]][1]

        vertical_tilt = HeadTopZ - HeadBotZ
        horizontal_tilt = SideL - SideR
    else:
        vertical_tilt = -1000
        horizontal_tilt = -1000
    
    return vertical_tilt, horizontal_tilt



def display_facemesh(image, results):
    face_landmarks = results.multi_face_landmarks[0]        # only face in list since max_num_faces=1
    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())
        
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path",
                    type=str)

    
    args = parser.parse_args()
    folder_path = args.folder_path
    

    # Constant
    right_eye = [[33, 133], [160, 144], [159, 145], [158, 153]] # right eye landmark positions
    left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]] # left eye landmark positions
    mouth = [[61, 291], [39, 181], [0, 17], [269, 405]] # mouth landmark coordinates

    # Nose idx
    NoseTop = [168, 6] # either one
    NoseBot = [4]
    # Head
    HeadTop = [10]
    HeadBot = [152]
    # Side of face
    Side_Left = [137]
    Side_Right = [454]
    # Nodding trackers
    Nod = NoseBot + HeadTop + HeadBot + Side_Left + Side_Right
    

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh


    for f in os.listdir(folder_path):
        if f.endswith('.mkv') or f.endswith('.mp4') or f.endswith('.avi'):
            print("\nAnalysing " + f)

            # For webcam input:
            drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

            cap = cv2.VideoCapture(folder_path + '/' + f)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fs = 60
            # print(total_frames)

            display = False

            with mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as face_mesh:
                with tqdm(total=total_frames) as pbar:
                    with open(folder_path + '/' + f[0:-4] + '_landmarks' +'.csv', 'w', newline='') as lm_f:
                        with open(folder_path + '/' + f[0:-4] + '_features' +'.csv', 'w', newline='') as f:
                            writer = csv.writer(f)
                            lm_writer = csv.writer(lm_f)

                            lm_col_headers = []
                            for lm in range(1,479):
                                lm_col_headers.append(str(lm)+"_x")
                                lm_col_headers.append(str(lm)+"_y")
                                lm_col_headers.append(str(lm)+"_z")

                            # print(lm_col_headers)
                            writer.writerow(["Time(s)", "EAR", "MAR", "PUC", "MOE", "Vertical Tilt", "Horizontal Tilt"])
                            lm_writer.writerow(["Time(s)"] + lm_col_headers)
                            t = 0
                            while cap.isOpened():
                                success, image = cap.read()
                                if not success:
                                    print("Ignoring empty camera frame.")
                                    # If loading a video, use 'break' instead of 'continue'.
                                    break

                                # To improve performance, optionally mark the image as not writeable to
                                # pass by reference.
                                image.flags.writeable = False
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                results = face_mesh.process(image)


                                landmarks_positions = []
                                landmarks_positions_raw = []
                                if results.multi_face_landmarks:
                                    for _, data_point in enumerate(results.multi_face_landmarks[0].landmark):
                                            landmarks_positions.append([data_point.x, data_point.y, data_point.z]) # saving normalized landmark positions
                                    landmarks_positions = np.array(landmarks_positions)
                                    landmarks_positions_raw = np.copy(landmarks_positions)
                                    landmarks_positions[:, 0] *= image.shape[1]
                                    landmarks_positions[:, 1] *= image.shape[0]

                                    if display:
                                        for idx in range(len(Nod)):
                                            relative_x = int(landmarks_positions[Nod[idx]][0])
                                            relative_y = int(landmarks_positions[Nod[idx]][1])
                                            cv2.circle(image, (relative_x, relative_y), radius=3, color=(0, 0, 255), thickness=4)
                                        
                                        display_facemesh(image, results)                                                                    

                                

                                ear, mar, puc, moe = get_face_features()
                                vertical_tilt, horizontal_tilt = get_head_features()
                                # Vertical tilt: -ve = tilted towards bottom, +ve = tilted towards top
                                # Horizontal tilt: -ve = tilted towards left, +ve = tilted towards right

                                t += 1/fs
                                # print(landmarks_positions.shape)


                                try:
                                    landmarks_positions.shape[0]
                                except:
                                    landmarks_positions = np.array(landmarks_positions)
                                
                                lm_pos_list = []
                                for lm in range(landmarks_positions.shape[0]):
                                    lm_pos_list.append(landmarks_positions[lm, 0]/image.shape[1])
                                    lm_pos_list.append(landmarks_positions[lm, 1]/image.shape[0])
                                    lm_pos_list.append(landmarks_positions[lm, 2])

                                lm_writer.writerow([round(t,3)] + lm_pos_list)
                                writer.writerow([round(t,3), ear, mar, puc, moe, vertical_tilt, horizontal_tilt])
                                pbar.update(1)

                                if cv2.waitKey(1) & 0xFF == 27:
                                    break
                                    
            cap.release()

            # # convert features to pkl
            # df = pd.read_csv(folder_path + '/' + f[0:-4] + '_features' +'.csv')
            # df.to_pickle(folder_path + '/' + f[0:-4] + '_features' + ".pkl", compression='bz2')
            # # df = pd.read_pickle("DATA/test/test_landmarks" + ".pkl", compression='bz2')
            # print("Converted " + f + " to pkl (compression = bz2)")
            
            # # convert landmarks to pkl
            # df = pd.read_csv(folder_path + '/' + f[0:-4] + '_landmarks' +'.csv')
            # df.to_pickle(folder_path + '/' + f[0:-4] + '_landmarks' + ".pkl", compression='bz2')
            # # df = pd.read_pickle("DATA/test/test_landmarks" + ".pkl", compression='bz2')
            # print("Converted " + f + " to pkl (compression = bz2)")


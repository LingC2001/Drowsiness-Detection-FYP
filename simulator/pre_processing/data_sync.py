import numpy as np
import pandas as pd
import csv
import os 
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

class Syncer:
    def __init__(self, cap_time, sim_time, bci_time, steer_time, cap_filename=None, sim_filename=None, bci_filename=None, steer_filename=None):
        """
        :param cap_time: number of seconds recording starts in pre_sync recording
        :param sim_time: number of seconds recording starts in pre_sync recording
        :param bci_time: number of seconds recording starts in pre_sync recording
        :param steer_time: number of seconds recording starts in pre_sync recording
        :param bci_filename: str filename of the openbci file
        """
        self.fixed_setup_time = 20
        self.cap_time = cap_time
        self.sim_time = sim_time
        self.bci_time = bci_time
        self.steer_time = steer_time
        
        
        self.save_folder = "synced/"
        self.cap_filename = cap_filename
        self.sim_filename = sim_filename
        self.bci_filename = bci_filename
        self.steer_filename = steer_filename
        self.data_duration = 60*29.5



    def calculate_time_to_crop(self)-> list[float]:
        """
        Used to calculate number of seconds to crop for each file to sync the times

        :returns crop_times: list of number of seconds to crop from the beginning of each file

        """
        start_times = [self.cap_time, self.sim_time, self.bci_time, self.steer_time]
        crop_baseline = np.argmax(start_times)
        crop_times = [0, 0, 0, 0]

        for i in range(len(start_times)):
            crop_times[i] = (start_times[crop_baseline] - start_times[i]) + self.fixed_setup_time
        print(crop_times)
        return crop_times

    def crop_files(self, crop_times: list[float])->None:
        """
        Used to perform cropping of the files


        """
        if not os.path.exists(self.save_folder):
            print("new dir created")
            os.mkdir(self.save_folder)
        if self.cap_filename is not None:
            self.crop_cap(crop_times[0])
        if self.sim_filename is not None:
            self.crop_sim(crop_times[1])
        if self.bci_filename is not None:
            self.crop_bci(crop_times[2])
        if self.steer_filename is not None:
            self.crop_steer(crop_times[3])

    def crop_bci(self, crop_time: float)->None:
        """
        Given crop_time, number of seconds to crop from the front of bci file and make duration 29 min
        
        """
        df = pd.read_csv(self.bci_filename, header=None)

        first_unix_time = float(df.iloc[0].to_list()[0].split("\t")[-2])
        new_start_time = first_unix_time + crop_time
        new_end_time = new_start_time + self.data_duration
        
        with open(self.save_folder + 'bci.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Sample Index", "EXG Channel 0", "EXG Channel 1", "EXG Channel 2", "EXG Channel 3", "EXG Channel 4", "EXG Channel 5", "EXG Channel 6", 
                             "EXG Channel 7", "EXG Channel 8", "EXG Channel 9", "EXG Channel 10", "EXG Channel 11", "EXG Channel 12", "EXG Channel 13", "EXG Channel 14", 
                             "EXG Channel 15", "Accel Channel 0", "Accel Channel 1", "Accel Channel 2", "Other", "Other", "Other", "Other",
                             "Other", "Other", "Other", "Analog Channel 0", "Analog Channel 1", "Analog Channel 2", "Timestamp", "Other"])
            for i in range(df.shape[0]):
                row_data = df.iloc[i].to_list()[0].split("\t")
                unix_time = float(row_data[-2])
                if unix_time >= new_start_time and unix_time <= new_end_time:
                    writer.writerow(row_data)
        print("BCI file cropped")
    
    def crop_steer(self, crop_time: float)->None:
        """
        Given crop_time, number of seconds to crop from the front of steer_output file and make duration 29 min
        
        """
        df = pd.read_csv(self.steer_filename)
        with open(self.save_folder + 'steer.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time", "steering", "acc", "brake"])
            for i in range(df.shape[0]):
                row_data = df.iloc[i].to_list()
                time = float(row_data[0])
                if time >= crop_time and time <= crop_time+self.data_duration:
                    writer.writerow(row_data)
        print("Steer Output file cropped")

    def crop_cap(self, crop_time: float)->None:
        ffmpeg_extract_subclip(self.cap_filename, crop_time, crop_time+self.data_duration, targetname=self.save_folder+self.cap_filename[0:-4]+".mp4")
        print("Camera capture file cropped")

    def crop_sim(self, crop_time: float)->None:
        ffmpeg_extract_subclip(self.sim_filename, crop_time, crop_time+self.data_duration, targetname=self.save_folder+self.sim_filename[0:-4]+".mp4")
        print("Simulator recording file cropped")

    def sync(self)->None:

        crop_times = self.calculate_time_to_crop()
        self.crop_files(crop_times)

    
if __name__ == "__main__":
    CAP_START_TIME = 13+37/60
    SIM_START_TIME = 16+14/60
    BCI_START_TIME = 120+ 30+1/60
    STEER_START_TIME = 11 +2/60
    cap_filename = "cap_17_afternoon.mkv"
    sim_filename = "sim_17_afternoon.mkv"
    bci_filename = "BrainFlow-RAW_2023-07-15_17-39-02_0.csv"
    steer_filename = "steer_output.csv"

    syncer = Syncer(CAP_START_TIME, SIM_START_TIME, BCI_START_TIME, STEER_START_TIME, cap_filename, sim_filename, bci_filename, steer_filename)
    syncer.sync()

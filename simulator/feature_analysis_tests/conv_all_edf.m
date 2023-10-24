clear all; close all; clc;

source_folder_name = "bci_files\";
dest_folder_name = "bci_clean\";

sesh = ["a","m","n"];

% Cleaning up and filtering the data for conversion
for i = 1:17*3
    
    sesh_id = mod(i,3);
    if sesh_id==0
        sesh_id = 3;
    end

    % Obtaining the data
    if i==26 || i==27 || i==34 || i==35 || i==36
        continue;
    elseif ceil(i/3) ~= 9 || ceil(i/3) ~= 12
        file_dir = sprintf("%s%i%s.csv",source_folder_name,ceil(i/3),sesh(sesh_id));
        data_to_clean = readmatrix(file_dir);
    elseif i==25
        file_dir = sprintf("%s%i%s.csv",source_folder_name,9,sesh(mod(i,3)));
        data_to_clean = readmatrix(file_dir);
    end
    
    % Saving the data into separate matrices
    if i~=41
        ecg_data = data_to_clean(:,2);
        emg_data = data_to_clean(:,6);
        eeg_data = data_to_clean(:,10:15);
    else
        ecg_data = data_to_clean(:,6);
        emg_data = data_to_clean(:,2);
        eeg_data = data_to_clean(:,10:15);
    end
    data = [ecg_data,emg_data,eeg_data];

    % General Variables
    time = (0:4:length(data)*4-1)';  % Time vector
    N_ch = 8;    

    % Band-pass Filtering Paramaters
    fsamp = 125;                    % Sampling frequency
    tsample = 1/fsamp;              % Period of samples
    f_low = 0.5;                     % Cut frequency for high-pass filter
    f_high = 40;                     % Cut frequency for low-pass filter

    % Pre-processing with Bandpass Filter
    for j=1:N_ch
        cleaned_data(:,j)= bandpass_filter_8ch(data(:,j), fsamp, f_low, f_high);
    end

    % Saving the data
    dest_file_dir = sprintf("%s%i%s_cleaned.csv",dest_folder_name,ceil(i/3),sesh(sesh_id));
    writematrix(cleaned_data,dest_file_dir);

    clear cleaned_data;

end
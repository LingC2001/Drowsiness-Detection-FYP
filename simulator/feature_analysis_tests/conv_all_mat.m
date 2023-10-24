clear all; close all; clc;

source_folder_name = "bci_clean\";
dest_folder_name = "bci_mat\";

sesh = ["a","m","n"];


for i = 1:17*3
    
    sesh_id = mod(i,3);
    if sesh_id==0
        sesh_id = 3;
    end

    % Obtaining the data
    if i==26 || i==27 || i==34 || i==35 || i==36
        continue;
    elseif ceil(i/3) ~= 9 || ceil(i/3) ~= 12
        file_dir = sprintf("%s%i%s_cleaned.csv",source_folder_name,ceil(i/3),sesh(sesh_id));
        data_to_clean = readmatrix(file_dir);
    elseif i==25
        file_dir = sprintf("%s%i%s_cleaned.csv",source_folder_name,9,sesh(mod(i,3)));
        data_to_clean = readmatrix(file_dir);
    end
    
    dest_file_dir = sprintf("conv_%i%s.mat",ceil(i/3),sesh(sesh_id));
    save(dest_file_dir,"data_to_clean");

    clear data_to_clean;

end
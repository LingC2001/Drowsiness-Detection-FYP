clear all; close all; clc;

file = importdata("bci_clean\1a_cleaned.csv");

ecg = file(:,1);

plot(ecg);
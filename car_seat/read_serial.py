import serial
import cv2
import time

serialPort = serial.Serial(port = "COM4", baudrate=9600,
                           bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
serialString = ""                           # Used to hold data coming over UART


img = cv2.imread('serial.PNG')
with open('acc.txt', 'w') as f:
    cv2.imshow('serial', img)
    serialPort.flushInput()
    serialPort.flushOutput()
    while((cv2.waitKey(1) & 0xFF)!= ord('q')):

        # Wait until there is data waiting in the serial buffer
        if(serialPort.in_waiting > 0):

            # Read data out of the buffer until a carraige return / new line is found
            serialString = serialPort.readline().decode('Ascii')
            t = time.asctime( time.localtime(time.time()))

            f.write(t+'||')
            f.write(serialString)
            print(serialString)

            # Read data out of the buffer until a carraige return / new line is found
            serialString = serialPort.readline().decode('Ascii')
            t = time.asctime( time.localtime(time.time()))

# Removing empty lines in file
with open('acc.txt', 'r') as f:
    filelines = f.readlines()

with open('acc.txt', 'w') as f:
    for lines in filelines:
        if lines != "\n":
            f.write(lines)
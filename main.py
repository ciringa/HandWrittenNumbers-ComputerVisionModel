from ultralytics import YOLO
import cv2 as cv



video = cv.VideoCapture("videos/output.avi");

model = YOLO("Src dos weights");


import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser(description="Video to MHI")
parser.add_argument("input", type=str, help="Path to input file")
parser.add_argument("-tp", "--type", type=str, help="Weather to return MHI or MEI. Must be either 'mhi' or 'mei'")
parser.add_argument("-th", "--thresh", type=float, help="Set threshold")
parser.add_argument("-t", "--tau", type=int, help="Set tau value")
parser.add_argument("-d", "--decay_rate", type=float, help="Set decay rate") 
parser.add_argument("-f", "--fps", type=int, help="Set output fps")
parser.add_argument("-o", "--output", type=str, help="Path to output file")
args = parser.parse_args()

if not args.thresh:
    args.thresh = 0.05
if not args.type:
    args.type = 'mhi'
if not args.tau:
    args.tau = 255 
if not args.decay_rate:
    args.decay_rate = 4 
if not args.fps:
    args.fps = 30
if not args.output:
    args.output = "mhi.mp4"

cap = cv2.VideoCapture(args.input)  
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
grayscale = np.empty((frameCount, frameHeight, frameWidth), np.dtype("uint8")) 

fc = 0
ret = True
while (fc < frameCount and ret):  # считываем видео
    ret, buf = cap.read()
    grayscale[fc] = cv2.cvtColor(buf, cv2.COLOR_BGR2GRAY)
    fc += 1

output = np.zeros_like(grayscale, dtype="int16")
if args.type.lower() == "mhi":
    for i in range(1, len(grayscale) - 1):
        one_idx = np.where((np.abs(grayscale[i] / 255 - grayscale[i - 1] / 255) > args.thresh) == 1)
        zero_idx = np.where((np.abs(grayscale[i] / 255 - grayscale[i - 1] / 255) > args.thresh) == 0)
        output[i][one_idx] = args.tau 
        output[i][zero_idx] = np.max((output[i - 1][zero_idx] - args.decay_rate, np.zeros_like(output[i - 1][zero_idx])), axis=0) 
else:
    for i in range(1, len(grayscale) - 1):
        output[i] = np.abs(grayscale[i] / 255 - grayscale[i - 1] / 255) > args.thresh
    output *= 255
        
out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"MP4V"), args.fps, (frameWidth, frameHeight), 0)
for frame in output.astype("uint8"):
    out.write(frame)
out.release()
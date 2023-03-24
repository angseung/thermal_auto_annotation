import os
import cv2

# video_file_name = "standing_a_person"
# video_file_name = "standing_people"
# video_file_name = "long_take"
video_file_name = "wide"
# video_file_name = "lie"
capture = cv2.VideoCapture(f"videos/{video_file_name}.mkv")
frameNr = 0
target_dir = f"images/{video_file_name}"

if not os.path.isdir(target_dir):
    os.makedirs(target_dir, exist_ok=True)

xtl, ytl = 482, 257
w, h = 262, 202

while True:
    success, frame = capture.read()
    if success:
        if frameNr % 2 == 0:
            frame = frame[ytl : ytl + h, xtl : xtl + w, :]
            cv2.imwrite(f"{target_dir}/frame_%05d_1.jpg" % frameNr, frame)

    else:
        break

    frameNr = frameNr + 1

capture.release()

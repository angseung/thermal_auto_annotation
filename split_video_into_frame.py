import cv2

video_file_name = "standing_people"
capture = cv2.VideoCapture(f"videos/{video_file_name}.mkv")
frameNr = 0

xtl, ytl = 482, 257
w, h = 262, 202

while True:
    success, frame = capture.read()
    if success:
        if frameNr % 10 == 0:
            frame = frame[ytl : ytl + h, xtl : xtl + w, :]
            cv2.imwrite(f"images/{video_file_name}/frame_%05d.jpg" % frameNr, frame)

    else:
        break

    frameNr = frameNr + 1

capture.release()

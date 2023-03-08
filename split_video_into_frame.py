import cv2

capture = cv2.VideoCapture("videos/standing_a_person.mkv")
frameNr = 0

xtl, ytl = 482, 257
w, h = 262, 202

while True:
    success, frame = capture.read()
    if success:
        if frameNr % 10 == 0:
            frame = frame[ytl : ytl + h, xtl : xtl + w, :]
            cv2.imwrite("images/standing_a_person/frame_%05d.jpg" % frameNr, frame)

    else:
        break

    frameNr = frameNr + 1

capture.release()

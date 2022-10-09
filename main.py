import cv2
import time
import mediapipe as mp

# for opening webcam
cap = cv2.VideoCapture(0)

# for mediapipe hand solutions
mp_hands = mp.solutions.hands

# perform processing of the detected hands
# static_image_mode => False(video format) & True(image format)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5)
# for drawing hand connections and landmarks on the frame
mpDraw = mp.solutions.drawing_utils

# for counting the number of fingers raised
finger_Coord = [(8, 6), (12, 10), (16, 14), (20, 18)]
thumb_Coord = (4, 2)


ptime = 0
while True:
    # cap.read() returns the frame of the video
    success, image = cap.read()

    # frame rate calculation
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(image, f'FPS: {int(fps)}', (400, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # before sending it to mediapipe, make into RGB image
    RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # send image to mediapipe and store the landmarks returned in results variable
    results = hands.process(RGB_image)
    # multiLandMarks is the list of all (x,y) coordinates of the hands detected
    multiLandMarks = results.multi_hand_landmarks

    # if that list is not empty
    if multiLandMarks:
        handlist = []
        # iterate over each coordinate and mark it on the image
        for handLms in multiLandMarks:
            mpDraw.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)
            # convert the coordinates to pixels
            for lm in handLms.landmark:
                h, w, c = image.shape
                # method for coordinates->pixels
                cx, cy = int(lm.x*w), int(lm.y*h)
                # save it, will be used for making circles at the landmark points
                handlist.append((cx, cy))
            # draw circles at landmark points
            for point in handlist:
                cv2.circle(image, point, 10, (255, 255, 0), cv2.FILLED)

        # performing counting logic and displaying the count
        upcount = 0
        for coordinate in finger_Coord:
            # as in an image, y coordinate increases as we move down
            if handlist[coordinate[0]][1] < handlist[coordinate[1]][1]:
                upcount += 1
        if handlist[thumb_Coord[0]][0] > handlist[thumb_Coord[1]][0]:
            upcount += 1

        # display the number in the image frame
        cv2.putText(image, str(upcount), (150, 150),
                    cv2.FONT_HERSHEY_PLAIN, 12, (0, 255, 0), 12)

    cv2.imshow("Fingers Counting Algorithm", image)

    # display cam untill 'q' key pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

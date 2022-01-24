import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

Img_Count = 0;

while True:
    #ret is a boolean variable that returns true if the frame is available
    #frame is an image array vector captured based on the default frames per second defined explicitly or implicitly
    ret, frame = cap.read()

    roi = frame[100:300, 100:300]
    img = cv2.resize(roi, (50, 50))
    cv2.imshow('Sample', roi)

    if cv2.waitKey(10) & 0xFF == ord('s'):

        cv2.imwrite(str(Img_Count)+'.jpg', roi)
        print(Img_Count)
        Img_Count = Img_Count + 1

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('Done')
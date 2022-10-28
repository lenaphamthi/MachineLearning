import cv2

#gán video muốn play
vid_capture = cv2.VideoCapture('file.mp4')

while (vid_capture.isOpened()):
    # vid_capture.read() trả về một tuple,phần tử đầu là bool
    # và phần tử thứ 2 là frame
    ret, frame = vid_capture.read()
    if ret == True:
        cv2.imshow('vid', frame)
        # Tăng thêm số giây phát video
        key = cv2.waitKey(20)
        if key == ord('q'):
            break
    else:
        break
# Release video
vid_capture.release()
cv2.destroyAllWindows()
import cv2
import time

def get_image():
  video_capture = cv2.VideoCapture(0)

  if not video_capture.isOpened():
      raise Exception("Could not open video device")
  time.sleep(0.5)
  ret, frame = video_capture.read()
  time.sleep(0.5)
  video_capture.release()

  image = cv2.resize(frame, (64, 64))
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  cv2.imshow('image', image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  # print(type(image))
  # print(image.shape)
  return image
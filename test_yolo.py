from detectors.ppe_detector import PPEDetector
import cv2


detector = PPEDetector()


image = cv2.imread("test.jpg")


result = detector.detect(image)


result.show()
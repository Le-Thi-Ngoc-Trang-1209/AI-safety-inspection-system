import cv2

img = cv2.imread("test.jpg")

if img is None:
	print("Error")
else:
	print("OK")
	cv2.imshow("image", img)
	cv2.waitKey(0)
	print("OK")
	cv2.destroyAllWindows()

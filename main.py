import cv2
import matplotlib.pyplot as plt

# %matplotlib inline

i1 = cv2.imread("lay.jpg")
i2 = cv2.imread("temp1.jpg")

img1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

k_1, des_1 = sift.detectAndCompute(img1, None)
k_2, des_2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(des_1, des_2)
matches = sorted(matches, key=lambda x: x.distance)

img3 = cv2.drawMatches(img1, k_1, img2, k_2, matches[:50], img2, flags=2)
img3 = cv2.resize(img3, (1000, 1000))
cv2.imshow("Output", img3)
#cv2.resizeWindow('Output', 1000, 1000)
cv2.waitKey(0)
cv2.destroyAllWindows()

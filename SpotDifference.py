import cv2
import imutils
from skimage.metrics import structural_similarity as ssim

imageA = cv2.imread(r"D:\Work\Ennoventure\images\spongebob1.jpg", cv2.IMREAD_ANYCOLOR)
imageB = cv2.imread(r"D:\Work\Ennoventure\images\spongebob2.jpg", cv2.IMREAD_ANYCOLOR)

greyA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
greyB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

(score, diff) = ssim(greyA, greyB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for c in cnts:
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)
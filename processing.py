import cv2
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
import imutils
from PIL import Image
from StringIO import StringIO
import base64

class ImageProcessingManager:

    def __init__(self):
        colors = OrderedDict({
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            })

        self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
        self.colorNames = []

        for (i, (name, rgb)) in enumerate(colors.items()):
            self.lab[i] = rgb
            self.colorNames.append(name)

        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)


    def label(self, image, c):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.erode(mask, None, iterations=2)
        mean = cv2.mean(image, mask=mask)[:3]

        minDist = (np.inf, None)

        for (i, row) in enumerate(self.lab):
            d = dist.euclidean(row[0], mean)
            if d < minDist[0]:
                minDist = (d, i)

        return self.colorNames[minDist[1]]

    def processForColor(self, base_64_image):
        image = self.base64ToImage(base_64_image)
        image = self.whiteToBlack(image)
        resized = imutils.resize(image, width=300)
        ratio = image.shape[0] / float(resized.shape[0])

        blurred = cv2.GaussianBlur(resized, (9, 9), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
        median = cv2.medianBlur(thresh, 5)

        cnts = cv2.findContours(median.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        colorsWithPosition = []

        for c in cnts:
            M = cv2.moments(c)
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)

            color = self.label(lab, c)
            colorsWithPosition.append([cX, cY, color])

        return colorsWithPosition

    def base64ToImage(self, base64_string):
        sbuf = StringIO()
        sbuf.write(base64.b64decode(base64_string))
        pimg = Image.open(sbuf)
        return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

    def test(self, filename):
        img = open(filename, "rb").read()
        encoded = base64.b64encode(img)
        self.processForColor(encoded)

    def whiteToBlack(self, img):
        width, height, channels = img.shape

        for x in range(width):
            for y in range(height):
                if img[x, y, 0] >= 220 and img[x, y, 1] >= 220 and img[x, y, 2] >= 220:
                    img[x, y, 0] = 0
                    img[x, y, 1] = 0
                    img[x, y, 2] = 0

        return img

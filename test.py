from ai_picture_processor import AIPictureProcessor
from time import sleep

import os

ap = AIPictureProcessor()
img = "/root/DiffServer/assets/test/test.jpg"
ap.start_job(img)
while True:
	sleep(1)

#print(os.path.splitext(img)[-1])


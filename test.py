from ai_picture_processor import AIPictureProcessor
from time import sleep

import os

ap = AIPictureProcessor()
img = "/users/a/diff/assets/test/test.jpg"
# ap.start_job(img)

print(os.path.splitext(img)[-1])


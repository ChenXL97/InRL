import os

import cv2

path = f"{os.path.dirname(__file__)}/../video/pictures"
out_path = f"{os.path.dirname(__file__)}/../video"

num_pics = len(os.listdir(f"{path}"))
img = cv2.imread(f'{path}/0.png')
fps = 9
videoWriter = cv2.VideoWriter(f'{out_path}/TestVideo_dog.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps,
                              (img.shape[1], img.shape[0]), True)

for i in range(num_pics):
    img = cv2.imread(f"{path}/{i}.png")
    videoWriter.write(img)

videoWriter.release()
cv2.destroyAllWindows()

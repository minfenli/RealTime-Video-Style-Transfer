import torch 
import cv2
import numpy as np

class ReshapeTool():
    def __init__(self):
        self.record_H = 0
        self.record_W = 0

    def process(self, img):
        H, W, C = img.shape
        # cv2.imshow("style transfer", img)
        # print(img.shape)

        if self.record_H == 0 and self.record_W == 0:
            new_H = H + 128
            if new_H % 64 != 0:
                new_H += 64 - new_H % 64

            new_W = W + 128
            if new_W % 64 != 0:
                new_W += 64 - new_W % 64

            self.record_H = new_H
            self.record_W = new_W

        new_img = cv2.copyMakeBorder(img, 64, self.record_H-64-H,
                                          64, self.record_W-64-W, cv2.BORDER_REFLECT)
        # print(new_img.shape)
        # cv2.imshow("style transfer", new_img)
        return new_img

def resize(image, max_resolution=1000000):
    resolution = max(image.shape[0], image.shape[1])
    if resolution > max_resolution:
        ratio = (resolution // 240) + 1
        return cv2.resize(image, (image.shape[1]//ratio, image.shape[0]//ratio))
    return image

def frame_global_sample(frames):
    style_framework.clean()

    for frame in frames:
        style_framework.add(frame)
    
    style_framework.compute()

def numpy2tensor(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img.transpose((2, 0, 1))).float()

def transform_image(img):
    img = img.div_(255.0)
    return img.unsqueeze(0)

def transform_back_image(img):
    img = img[0].clamp(0, 1) # * 255
    return img

def get_fps(times):
    return np.round(1/np.mean(times), 2)

def get_camera_frame():
    count = 0
    cap = cv2.VideoCapture(0)

    while count != 10:
        flag, frame = cap.read()
        if flag:
            cap.release()
            return True, frame
        count += 1

    cap.release()
    return False, None
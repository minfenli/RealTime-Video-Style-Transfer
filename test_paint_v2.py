from configparser import Interpolation
from style_transfer.test.framework import Stylization
from mat.model import MattingNetwork
import numpy as np 
import torch
import os
import cv2
from utils import *

import importlib
from tqdm import tqdm 
import matplotlib.pyplot as plt

## --------------------------------------------
##  Parameters for PyTorch
## --------------------------------------------

device = 'cuda'

## --------------------------------------------
##  Parameters for the matting model
## --------------------------------------------

model_type = 'mobilenetv3' # 'resnet50'
mat_checkpoint_path = './mat/checkpoints/rvm_mobilenetv3.pth' # 'rvm_resnet50'

## --------------------------------------------
##  Parameters for the inpainting model
## --------------------------------------------

inpaint_model_type = 'e2fgvi'
inpaint_model_ckpt = 'inpainting/release_model/E2FGVI-CVPR22.pth'

## --------------------------------------------
##  Initialize the inpainting model
## --------------------------------------------

# net = importlib.import_module('inpainting.model.' + inpaint_model_type)
# inpaint_model = net.InpaintGenerator().to(device)
# data = torch.load(inpaint_model_ckpt, map_location=device)
# inpaint_model.load_state_dict(data)
# print(f'Loading inpaint model from: {inpaint_model_ckpt}')

inpaint_size = (240, 432)
neighbor_stride = 2 

## --------------------------------------------
##  Parameters for the matting model
## --------------------------------------------

model_type = 'mobilenetv3' # 'resnet50'
mat_checkpoint_path = './mat/checkpoints/rvm_mobilenetv3.pth' # 'rvm_resnet50'

## --------------------------------------------
##  Initialize the matting model
## --------------------------------------------

matting_model = MattingNetwork(model_type).eval().to(device)
matting_model.load_state_dict(torch.load(mat_checkpoint_path, map_location=device))

print('Loaded matting model "{}"'.format(mat_checkpoint_path))

## --------------------------------------------
##  Parameters for the style transfer model
## --------------------------------------------

style_img = "./style_transfer/inputs/styles/" + "starry_night.jpg"#  "The_Great_Wave_off_Kanagawa.jpg" #"starry_night.jpg" # "pencil.png" # "mosaic_2.jpg"
style_checkpoint_path = "./style_transfer/test/Model/style_net-TIP-final.pth"
style_img_resize_ratio = 1

use_Global = False

## --------------------------------------------
##  Initialize the style transfer model
## --------------------------------------------

style = cv2.imread(style_img)
style = cv2.resize(style, (style.shape[1]//style_img_resize_ratio, style.shape[0]//style_img_resize_ratio), cv2.INTER_AREA)
style_fname = os.path.split(style_img)[1]
print('Opened style image "{}"'.format(style_fname))

style_framework = Stylization(style_checkpoint_path, device, use_Global)
style_framework.prepare_style(style)

print('Loaded style framework "{}"'.format(style_checkpoint_path))

## --------------------------------------------
##  Parameters for processing
## --------------------------------------------

camera_resize_ratio = 1               # downsampling for a faster computing speed

sample_frames = 20                    # not necessary if use_Global==False
sample_frequency = 1                  # not necessary if use_Global==False

refresh_background_momentum = 0.8     # background = "momentum" * old_background + (1-"momentum") * new_background
refresh_background_frequency = 1      # frequency for updating the background

strict_alpha = False                 # alpha 0~1 values will be round to 0 or 1
inverse = False                       # inverse style transfer from background to people


## --------------------------------------------
##  Functions for processing
## --------------------------------------------

bgr = torch.tensor([.6, 1, .47]).view(3, 1, 1).cuda()  # Green background.
rec = [None] * 4                                       # Initial recurrent states.
downsample_ratio = 1  

reshape = ReshapeTool()

def frame_matting(frame):
    global bgr, rec, downsample
    with torch.no_grad():
        # Transform images into tensors
        frame = numpy2tensor(frame).to(device)
        src = transform_image(frame)
        
        fgr, pha, *rec = matting_model(src.cuda(), *rec, downsample_ratio)  # Cycle the recurrent states.
        # com = fgr.squeeze(0) * pha + bgr.squeeze(0) * (1 - pha)           # Composite to green background. 
        
        frame_result = transform_back_image(pha)
        return frame_result.cpu().numpy().astype('float16').transpose((1, 2, 0))

def frame_style_transfer(frame):
    frame = resize(frame[..., ::-1])

    # Crop the image
    H,W,C = frame.shape
    new_input_frame = reshape.process(frame)

    # Stylization
    styled_input_frame = style_framework.transfer(new_input_frame)

    # Crop the image back
    styled_input_frame = styled_input_frame[64:64+H,64:64+W,:]

    # cast as unsigned 8-bit integer (not necessarily needed)
    styled_input_frame = styled_input_frame.astype('uint8')

    return styled_input_frame[..., ::-1]

def frame_inpainting(frame, mask):

    masked_imgs = (frame * (1 - mask)).astype('float32')
    padding_imgs = np.zeros(masked_imgs.shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 3))
    kernel_sub = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    for i in range(3):
        padding_imgs[:, :, i] = cv2.dilate(masked_imgs[: ,:, i], kernel, iterations=1)
        padding_imgs[:, :, i] = cv2.dilate(padding_imgs[:, :, i] * mask[:, :, 0], kernel_sub, iterations=1)
    
    result = masked_imgs + padding_imgs * mask 
    blur_kernel = np.ones((5, 5), np.float32) / 25
    blur_imgs = cv2.filter2D(result, -1, blur_kernel)
    result = masked_imgs + blur_imgs * mask

    return result 

## --------------------------------------------
##  Call this function for the use
## --------------------------------------------

def run():
    global use_Global, resize_ratio, sample_frames, sample_frequency, refresh_background_momentum, refresh_background_frequency, strict_alpha, inverse

    frame_counter = 0
    frame_buffer = []

    cap = cv2.VideoCapture(0)
    print("Open Camera")

    # try:
    while True:
        flag,frame=cap.read()
        if flag:
            # image preprocessing

            frame_resize = cv2.resize(frame, (frame.shape[1]//camera_resize_ratio, frame.shape[0]//camera_resize_ratio), cv2.INTER_AREA)
            image_rgb_np = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2RGB)


            # calculate the alpha per pixel for matting
            # if frame_counter > 0:
            #     image_int, ori_int = image_rgb_np.astype('int'), ori.astype('int')
            #     diff = np.abs(image_int - ori_int)
            #     r_mask = (diff[:, :, 0]>0) & (diff[:, :, 0]<3)
            #     g_mask = (diff[:, :, 1]>0) & (diff[:, :, 1]<3)
            #     b_mask = (diff[:, :, 2]>0) & (diff[:, :, 2]<3)
            #     t_mask = (r_mask | g_mask | b_mask) * (~(r_mask & g_mask & b_mask))
            #     image_rgb_np[t_mask] = ori[t_mask]
            #     noise = np.abs(image_int - ori_int).astype('uint8')
            #     cv2.imshow("difference", t_mask.astype('uint8')*255)
            #     cv2.imshow("noise", noise.astype('uint8')*255)

                # image_rgb_np[np.abs(image_rgb_np - ori) < 5] = ori[np.abs(image_rgb_np - ori) < 5]
                # diff = np.abs(image_rgb_np - ori).astype('uint8')
                # image_rgb_np = image_rgb_np.astype('uint8')

                # diff = np.abs(image_rgb_np - ori)/5.
                # diff[diff > 1] = 1
                # image_rgb_np = ori * (1-diff) + image_rgb_np * (diff)
            alpha = frame_matting(image_rgb_np)
            
            if strict_alpha:
                alpha[alpha>0.1] = 1
                # alpha = alpha.round()

            # extract the foreground and the background

            foreground = (image_rgb_np * alpha).astype('uint8')

            if inverse:
                background = (image_rgb_np * (1-alpha)).astype('uint8')
            else:
                if frame_counter % refresh_background_frequency == 0:
                    if frame_counter == 0:
                        background = frame_inpainting(image_rgb_np, alpha).astype('uint8')
                        # background = (image_rgb_np * (1-alpha)).astype('uint8')
                        old_alpha = alpha
                    else:
                        background = frame_inpainting(image_rgb_np, alpha).astype('uint8')
                        old_alpha = alpha
                        # background = ((background*refresh_background_momentum + image_rgb_np * (1-alpha) * (1-refresh_background_momentum))).astype('uint8')

            # initialize the global features if "use_Global"

            if use_Global and frame_counter == 0:
                if inverse:
                    frame_global_sample([foreground])
                else:
                    frame_global_sample([background])
            
            # # Dilating painting 
            # frame_list = np.concatenate([image_rgb_np[None, ...], image_rgb_np[None, ...]])
            # mask_list = np.concatenate([alpha[None, ...], alpha[None, ...]])
            # painted_image = frame_inpainting(frame_list, mask_list)
            # background = painted_image.astype('uint8')

            # fuse the origin frame with the style transfer result

            if inverse:
                transfer_result = (frame_style_transfer(foreground)*alpha + background).astype('uint8')
            else:
                transfer_result = (frame_style_transfer(background)*(1-alpha) + foreground).astype('uint8')
                # transfer_result = frame_style_transfer(background)
                # transfer_result = background
            
            # display the frame

            image_bgr_np=cv2.cvtColor(transfer_result, cv2.COLOR_RGB2BGR)
            show_background=cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
            cv2.imshow("style transfer", image_bgr_np)
            cv2.imshow("original background", show_background)

            # store the frame and calculate the global features if "use_Global" 
            if use_Global and frame_counter%sample_frequency == 0:
                if inverse:
                    frame_buffer.append(foreground)
                else:
                    frame_buffer.append(background)
                
                if len(frame_buffer) == sample_frames:
                    print("Calculate Global")
                    frame_global_sample(frame_buffer)
                    del frame_buffer[:]

            ori = image_rgb_np
            frame_counter += 1
        else:
            print("Something went wrong on the camera")
            break
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    # except:
    #     print(1111)
    #     cap.release()
    #     cv2.destroyAllWindows()

if __name__ == '__main__':
    run()
from style_transfer.test.framework import Stylization
from mat.model import MattingNetwork
import torch
import os
import cv2
from utils import *

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
##  Initialize the matting model
## --------------------------------------------

matting_model = MattingNetwork(model_type).eval().to(device)
matting_model.load_state_dict(torch.load(mat_checkpoint_path, map_location=device))

print('Loaded matting model "{}"'.format(mat_checkpoint_path))

## --------------------------------------------
##  Parameters for the style transfer model
## --------------------------------------------

style_img = "./style_transfer/inputs/styles/" + "pencil.png"#  "The_Great_Wave_off_Kanagawa.jpg" #"starry_night.jpg" # "pencil.png" # "mosaic_2.jpg"
style_checkpoint_path = "./style_transfer/test/Model/style_net-TIP-final.pth"
style_img_resize_ratio = 2

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

strict_alpha = False                  # alpha 0~1 values will be round to 0 or 1
inverse = False                       # inverse style transfer from background to people


frame_counter = 0
frame_buffer = []

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

## --------------------------------------------
##  Call this function for the use
## --------------------------------------------

def run():
    global use_Global, resize_ratio, frame_counter, frame_buffer, sample_frames, sample_frequency, refresh_background_momentum, refresh_background_frequency, strict_alpha, inverse

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
            
            alpha = frame_matting(image_rgb_np)
            
            if strict_alpha:
                alpha = alpha.round()


            # extract the foreground and the background
            
            foreground = (image_rgb_np * alpha).astype('uint8')
            
            if frame_counter % refresh_background_frequency == 0:
                if frame_counter == 0:
                    background = (image_rgb_np * (1-alpha)).astype('uint8')
                else:
                    background = ((background*refresh_background_momentum + image_rgb_np * (1-alpha) * (1-refresh_background_momentum))).astype('uint8')

            # initialize the global features if "use_Global"

            if use_Global and frame_counter == 0:
                if inverse:
                    frame_global_sample([foreground])
                else:
                    frame_global_sample([background])

            # fuse the origin frame with the style transfer result

            if inverse:
                transfer_result = (frame_style_transfer(foreground)*alpha + background).astype('uint8')
            else:
                transfer_result = (frame_style_transfer(background)*(1-alpha) + foreground).astype('uint8')
            
            # display the frame

            image_bgr_np=cv2.cvtColor(transfer_result, cv2.COLOR_RGB2BGR)
            cv2.imshow("style transfer", image_bgr_np)

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
from style_transfer.test.framework import Stylization
from mat.model import MattingNetwork
import torch
import os
import cv2
from utils import *
from skimage.metrics import structural_similarity as ssim_metric
import time 
from skimage import transform

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

strict_alpha = False                  # alpha 0~1 values will be round to 0 or 1
inverse = False                       # inverse style transfer from background to people
restore_foreground_resolution = True  # restore resolution of foreground back to carema input size

denoise = True                        # doing denoise that ignore small pixel value change
inpaint = True                        # doing inpaint for less ringing effects or artifacts

style_transfer = True                 # only matting if style_transfer = False

print_fps = False                     # print fps

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
    global use_Global, resize_ratio, sample_frames, sample_frequency, strict_alpha, inverse, restore_foreground_resolution, denoise, style_transfer, inpaint, print_fps

    frame_counter = 0
    frame_buffer = []
    computing_times = []

    if camera_resize_ratio == 1:
        restore_foreground_resolution = False

    cap = cv2.VideoCapture(0)
    print("Open Camera")

    try:
        while True:
            flag,frame=cap.read()
            if flag:
                start = time.time()

                # image preprocessing

                frame_resize = cv2.resize(frame, (frame.shape[1]//camera_resize_ratio, frame.shape[0]//camera_resize_ratio), cv2.INTER_AREA)
                image_rgb_np = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2RGB)

                # denoise filter

                if denoise:
                    if frame_counter != 0:
                        diff = np.sum(np.abs(image_rgb_np.astype('int')-image_rgb_np_old.astype('int')), axis = 2)/24.
                        diff[diff > 1] = 1
                        diff = diff**4
                        diff = np.expand_dims(diff, axis=-1)
                        image_rgb_np = image_rgb_np_old * (1-diff) + image_rgb_np * diff
                        image_rgb_np = image_rgb_np.astype("uint8")

                    image_rgb_np_old = image_rgb_np



                # calculate the alpha per pixel for matting
                
                alpha = frame_matting(image_rgb_np)

                # cv2.imshow("alpha", (alpha*255).astype('uint8'))
                
                if strict_alpha:
                    alpha = alpha.round()


                # extract the foreground and the background
                
                foreground = (image_rgb_np * alpha).astype('uint8')

                if not style_transfer:
                    # foreground as the matting result     
                    if restore_foreground_resolution:
                        alpha_origin_size = transform.resize(alpha, (frame.shape[0], frame.shape[1]))
                        image_rgb_np_origin_size = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        transfer_result = (alpha_origin_size*image_rgb_np_origin_size).astype('uint8')
                    else:
                        transfer_result = foreground
                else:
                    if inpaint and not inverse:
                        background = frame_inpainting(image_rgb_np, alpha).astype('uint8')
                    else:
                        background = (image_rgb_np * (1-alpha)).astype('uint8')

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
                        if restore_foreground_resolution:
                            transfer_result = (frame_style_transfer(background)*(1-alpha)).astype('uint8')
                            transfer_result = cv2.resize(transfer_result, (frame.shape[1], frame.shape[0]), cv2.INTER_AREA)
                            image_rgb_np_origin_size = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            alpha_origin_size = transform.resize(alpha, (frame.shape[0], frame.shape[1]))
                            transfer_result = (transfer_result + alpha_origin_size*image_rgb_np_origin_size).astype('uint8')
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

                # calculate fps
                if print_fps:
                    end = time.time()
                    computing_times.append(end - start)
                    if len(computing_times) > 10:
                        computing_times.pop(0)
                    print(get_fps(computing_times))

            else:
                print("Something went wrong on the camera")
                break
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()
    except:
        print("Exception!!")
        cap.release()
        cv2.destroyAllWindows()


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


def run_virtual_camera(device):
    global use_Global, resize_ratio, sample_frames, sample_frequency, strict_alpha, inverse, restore_foreground_resolution, denoise, style_transfer, inpaint, print_fps

    import pyvirtualcam

    flag, frame = get_camera_frame()

    if not flag:
        print("Error when getting size of a frame")
        return

    if restore_foreground_resolution and style_transfer:
        width, height = frame.shape[1], frame.shape[0]
    else:
        width, height = frame.shape[1]//camera_resize_ratio, frame.shape[0]//camera_resize_ratio

    with pyvirtualcam.Camera(width=width, height=height, fps=20, device=device) as cam:
        print(f'Using virtual camera: {cam.device}')

        cap = cv2.VideoCapture(0)

        frame_counter = 0
        frame_buffer = []
        computing_times = []

        if camera_resize_ratio == 1:
            restore_foreground_resolution = False

        print("Open Camera")

        try:
            while True:
                flag,frame=cap.read()
                if flag:
                    start = time.time()

                    # image preprocessing

                    frame_resize = cv2.resize(frame, (frame.shape[1]//camera_resize_ratio, frame.shape[0]//camera_resize_ratio), cv2.INTER_AREA)
                    image_rgb_np = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2RGB)

                    # denoise filter

                    if denoise:
                        if frame_counter != 0:
                            diff = np.sum(np.abs(image_rgb_np.astype('int')-image_rgb_np_old.astype('int')), axis = 2)/24.
                            diff[diff > 1] = 1
                            diff = diff**4
                            diff = np.expand_dims(diff, axis=-1)
                            image_rgb_np = image_rgb_np_old * (1-diff) + image_rgb_np * diff
                            image_rgb_np = image_rgb_np.astype("uint8")

                        image_rgb_np_old = image_rgb_np



                    # calculate the alpha per pixel for matting
                    
                    alpha = frame_matting(image_rgb_np)

                    # cv2.imshow("alpha", (alpha*255).astype('uint8'))
                    
                    if strict_alpha:
                        alpha = alpha.round()


                    # extract the foreground and the background
                    
                    foreground = (image_rgb_np * alpha).astype('uint8')

                    if not style_transfer:
                        # foreground as the matting result 
                        if restore_foreground_resolution:
                            alpha_origin_size = transform.resize(alpha, (frame.shape[0], frame.shape[1]))
                            image_rgb_np_origin_size = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            transfer_result = (alpha_origin_size*image_rgb_np_origin_size).astype('uint8')
                        else:
                            transfer_result = foreground
                    else:
                        if inpaint and not inverse:
                            background = frame_inpainting(image_rgb_np, alpha).astype('uint8')
                        else:
                            background = (image_rgb_np * (1-alpha)).astype('uint8')

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
                            if restore_foreground_resolution:
                                transfer_result = (frame_style_transfer(background)*(1-alpha)).astype('uint8')
                                transfer_result = cv2.resize(transfer_result, (frame.shape[1], frame.shape[0]), cv2.INTER_AREA)
                                image_rgb_np_origin_size = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                alpha_origin_size = transform.resize(alpha, (frame.shape[0], frame.shape[1]))
                                transfer_result = (transfer_result + alpha_origin_size*image_rgb_np_origin_size).astype('uint8')
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

                    # calculate fps
                    if print_fps:
                        end = time.time()
                        computing_times.append(end - start)
                        if len(computing_times) > 10:
                            computing_times.pop(0)
                        print(get_fps(computing_times))

                    cam.send(transfer_result)
                    cam.sleep_until_next_frame()

                else:
                    print("Something went wrong on the camera")
                    break
                
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
                
        except:
            print("Exception!!")
            cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='RealTime Style Transfer from Camera Inputs')
    parser.add_argument("--virtual_camera", help="Send the results to the virtual camera", action="store_true")
    args = parser.parse_args()
    if args.virtual_camera:
        run_virtual_camera('/dev/video2')
    else:
        run()
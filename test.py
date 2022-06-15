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

style_img = "./style_transfer/inputs/styles/" + "mosaic_2.jpg"#  "The_Great_Wave_off_Kanagawa.jpg" #"starry_night.jpg" # "pencil.png" # "mosaic_2.jpg"
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

style_framework = Stylization(style_checkpoint_path, device=="cuda", use_Global)
style_framework.prepare_style(style)

print('Loaded style framework "{}"'.format(style_checkpoint_path))

## --------------------------------------------
##  Parameters for processing
## --------------------------------------------

camera_resize_ratio = 2               # downsampling for a faster computing speed

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

bgr = torch.tensor([.6, 1, .47]).view(3, 1, 1).cuda() if device=="cuda" else torch.tensor([.6, 1, .47]).view(3, 1, 1) # Green background.
rec = [None] * 4                                       # Initial recurrent states.
downsample_ratio = 1  

reshape = ReshapeTool()

def frame_matting(frame):
    global bgr, rec, downsample
    with torch.no_grad():
        # Transform images into tensors
        frame = numpy2tensor(frame).to(device)
        src = transform_image(frame)
        
        fgr, pha, *rec = matting_model(src.cuda() if device=="cuda" else src, *rec, downsample_ratio)  # Cycle the recurrent states.
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

frame_counter = None

def frame_processing(frame, init=False, rgb=False):
    global use_Global, resize_ratio, sample_frames, sample_frequency, strict_alpha, inverse, camera_resize_ratio, restore_foreground_resolution, denoise, style_transfer, inpaint, print_fps

    global image_rgb_np_old, frame_counter, frame_buffer, computing_times

    if init or not frame_counter:
        if camera_resize_ratio == 1:
            restore_foreground_resolution = False
        
        frame_counter = 0
        frame_buffer = []
        computing_times = []

    start = time.time()

    # image preprocessing

    frame_resize = cv2.resize(frame, (frame.shape[1]//camera_resize_ratio, frame.shape[0]//camera_resize_ratio), cv2.INTER_AREA)
    image_rgb_np = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2RGB)

    # denoise filter

    if denoise:
        if frame_counter != 0:
            diff = np.sum(np.abs(image_rgb_np.astype('int')-image_rgb_np_old.astype('int')), axis = 2)/24.
            diff[diff > 1] = 1
            # diff = diff**4
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
            if restore_foreground_resolution:
                transfer_result = (frame_style_transfer(foreground)*alpha).astype('uint8')
                transfer_result = cv2.resize(transfer_result, (frame.shape[1], frame.shape[0]), cv2.INTER_AREA)
                image_rgb_np_origin_size = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                alpha_origin_size = transform.resize(alpha, (frame.shape[0], frame.shape[1]))
                transfer_result = (transfer_result + (1-alpha_origin_size)*image_rgb_np_origin_size).astype('uint8')
            else:
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

    # store the frame and calculate the global features if "use_Global" 
    if style_transfer and use_Global and frame_counter%sample_frequency == 0:
        if inverse:
            frame_buffer.append(foreground)
        else:
            frame_buffer.append(background)
        
        if len(frame_buffer) == sample_frames:
            print("Calculate Global")
            frame_global_sample(frame_buffer)
            del frame_buffer[:]

    # calculate fps
    if print_fps:
        end = time.time()
        computing_times.append(end - start)
        if len(computing_times) > 10:
            computing_times.pop(0)
        print(get_fps(computing_times))

    frame_counter += 1
    
    return transfer_result if rgb else cv2.cvtColor(transfer_result, cv2.COLOR_RGB2BGR)

## --------------------------------------------
##  Call this function for the use
## --------------------------------------------

def run():
    global use_Global, resize_ratio, sample_frames, sample_frequency, strict_alpha, inverse, restore_foreground_resolution, denoise, style_transfer, inpaint, print_fps

    cap = cv2.VideoCapture(0)
    print("Open Camera")

    try:
        while True:
            flag,frame=cap.read()
            if flag:
                start = time.time()

                transfer_result = frame_processing(frame)
                    
                # display the frame
                cv2.imshow("style transfer", transfer_result)

            else:
                print("Something went wrong on the camera")
                break
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()
    except:
        print("Exception!!")
        cap.release()
        cv2.destroyAllWindows()

def run_virtual_camera(device):
    global camera_resize_ratio, restore_foreground_resolution, style_transfer

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

        print("Open Camera")

        try:
            while True:
                flag,frame=cap.read()
                if flag:
                    start = time.time()

                    transfer_result = frame_processing(frame, rgb=True)
                        
                    # display the frame
                    cv2.imshow("style transfer", transfer_result)

                    cam.send(transfer_result)
                    cam.sleep_until_next_frame()

                else:
                    print("Something went wrong on the camera")
                    break
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
                
        except:
            print("Exception!!")
            cap.release()
            cv2.destroyAllWindows()

def transfer_video(input_file_path, out_file_path):
    global use_Global, resize_ratio, sample_frames, sample_frequency, strict_alpha, inverse, restore_foreground_resolution, denoise, style_transfer, inpaint, print_fps

    import imageio
    import moviepy.editor as mp
    from tqdm import tqdm

    # Read input video
    video_fname = os.path.split(input_file_path)[1]
    if not os.path.exists(input_file_path):
        exit('Input video %s does not exists (typo on your path?)' % (input_file_path))
    video = imageio.get_reader(input_file_path)
    fps = video.get_meta_data()['fps']
    print('Opened input video "{}" for style transfer (fps = {})'.format(video_fname, fps))

    my_clip = mp.VideoFileClip(input_file_path)
    audio_track = my_clip.audio
    if not audio_track:
        print('No audio found from input video')
    else:
        audio_sampling_freq = audio_track.fps
        audio_chs = audio_track.nchannels
        print('Opened the audio of the input video fps = {} Hz, number of channels = {}'.format(audio_sampling_freq, audio_chs))

    writer = imageio.get_writer("./temp.mp4", fps=fps)

    try:
        print(f'Style transfer for the file: {input_file_path}')
        for i, frame in tqdm(enumerate(video)):
            print(frame.shape)
            image_rgb_np = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            transfer_result = frame_processing(image_rgb_np)
            cv2.imshow("style transfer", transfer_result)
            writer.append_data(transfer_result[..., ::-1])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        writer.close()
    except:
        print("Exception while style transfer!!")
        writer.close()

        
    if audio_track:
        try:
            saved_mp4_read_back = mp.VideoFileClip("./temp.mp4")
            final_clip = saved_mp4_read_back.set_audio(audio_track)
            final_clip.write_videofile(out_file_path, fps=fps, verbose=False, logger=None)
            remove_status = os.remove("./temp.mp4")
        except:
            print("Exception while adding the audio back!!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='RealTime Style Transfer from Camera Inputs')
    parser.add_argument("--virtual_camera", help="Send the results to the virtual camera", action="store_true")
    parser.add_argument("--process_video", help="Load the video and store the result", action="store_true")
    parser.add_argument("--input", help="Input file location is required if \"process_video\"", type=str, default=None)
    parser.add_argument("--output", help="Output file location is required if \"process_video\"", type=str, default=None)
    args = parser.parse_args()
    if args.virtual_camera:
        run_virtual_camera('/dev/video2')
    elif args.process_video:
        if args.input and args.output:
            transfer_video(args.input, args.output)
        else:
            print("Input file location and output file location are required.")
    else:
        run()
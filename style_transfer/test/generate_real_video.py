import cv2
import glob
import os
import scipy.io as scio
import numpy as np
import random
import time
from datetime import timedelta

from tqdm import tqdm

import argparse
import torch
import imageio
import moviepy.editor as mp


from framework import Stylization

## -------------------
##  Parameters

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


# Path of the checkpoint (please download and replace the empty file)
# i.e. your pretrained model (weights) by the authors of the paper
checkpoint_path = "./Model/style_net-TIP-final.pth"

# Device settings, use cuda if available

# device = 'cpu' if args.cpu else 'cuda'

# The proposed Sequence-Level Global Feature Sharing
use_Global = True

# Where to save the results
results_base = '../results'
if not os.path.exists(results_base):
    os.mkdir(results_base)

result_videos_path = os.path.join(results_base, 'video')
if not os.path.exists(result_videos_path):
    os.mkdir(result_videos_path)


## -------------------
##  Tools

def read_img(img_path):
    return cv2.imread(img_path)


class ReshapeTool():
    def __init__(self):
        self.record_H = 0
        self.record_W = 0

    def process(self, img):
        H, W, C = img.shape

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
        return new_img


def resize(image, max_resolution=1000000):
    resolution = max(image.shape[0], image.shape[1])
    if resolution > max_resolution:
        ratio = (resolution // 240) + 1
        return cv2.resize(image, (image.shape[1]//ratio, image.shape[0]//ratio))
    return image

# def cvt_b(image, resize=None):
#     image[..., 2], image[..., 1], image[..., 0] = image[..., 0], image[..., 1], image[..., 2]
#     if False and resize:
#         image = cv2.resize(image, (image.shape[1]//resize, image.shape[0]//resize), interpolation=cv2.INTER_AREA)
#     return image

## -------------------
##  Preparation

def process_video(style_img, input_video, interval = 8, write_frames_to_disk = False, force_on_CPU = False):

    start_time_process = time.monotonic()

    # Read style image
    if not os.path.exists(style_img):
        exit('Style image %s does not exist (typo on your path?)'%(style_img))
    style = cv2.imread(style_img)
    style_fname = os.path.split(style_img)[1]
    print('Opened style image "{}"'.format(style_fname))

    # Read input video
    video_fname = os.path.split(input_video)[1]
    if not os.path.exists(input_video):
        exit('Input video %s does not exists (typo on your path?)' % (input_video))
    video = imageio.get_reader(input_video)
    fps = video.get_meta_data()['fps']
    print('Opened input video "{}" for style transfer (fps = {})'.format(video_fname, fps))

    # TODO! could be nicer to use just one library for image and audio data
    # https://towardsdatascience.com/extracting-audio-from-video-using-python-58856a940fd
    my_clip = mp.VideoFileClip(input_video)
    audio_track = my_clip.audio
    if not audio_track:
        print('No audio found from input video')
    else:
        audio_sampling_freq = audio_track.fps
        audio_chs = audio_track.nchannels
        print('Opened the audio of the input video f_sampling = {} Hz, number of channels = {}'.format(
            audio_sampling_freq, audio_chs))


    # TODO! modify here if you do not like the output filename convention
    name = 'ReReVST-' + ('' if use_Global else 'no-global-') + style_fname + '-' + video_fname

    # Build model
    start_time = time.monotonic()
    # TODO! Here you are reading in the pretrained weights from "checkpoint_path", and you could add noise to the
    #  pretrained weights, either once when reading them, or in the loop when stylizing images if you just want glitchy
    #  noise output and the output does not have to be that realistic
    cuda = torch.cuda.is_available()
    if force_on_CPU:
        print('Force computations to be done with CPU instead of GPU (e.g. you have more system RAM than GPU RAM,'
              'and you are willing to wait a bit more for computations)')
        cuda = False

    # TODO! you could try to accelerate this with simply making networkdeav fp16?
    framework = Stylization(checkpoint_path, cuda, use_Global)
    framework.prepare_style(style)
    end_time = time.monotonic()
    print('Stylization Model built in {}'.format(timedelta(seconds=end_time - start_time)))

    # Build tools
    reshape = ReshapeTool()

    ## -------------------
    ##  Inference

    # TODO! isn't there a way to just get the number of frames without this?
    for i, frame in enumerate(video):
        frame_num = i+1
    print('Number of frames in the input video = {} (length {:.2f} seconds)'.format(frame_num, frame_num/fps))

    # Prepare for proposed Sequence-Level Global Feature Sharing
    if use_Global:

        start_time = time.monotonic()
        print('Preparations for Sequence-Level Global Feature Sharing')
        framework.clean()
        sample_sum = (frame_num-1)//interval # increase interval if you run out of CUDA mem, e.g. to 16
        # TODO! it is actually the number of frames used (sample_sum), so you could do autocheck for this
        #  based on your hardware, and the processed video, and automatically increase the interval

        # get frame indices to be used
        indices = list()
        print('Using a total of {} frames to do global feature sharing (trying to use too many might result memory running out)'.format(sample_sum))
        for s in range(sample_sum):
            i = s * interval
            indices.append(i)
            # print(' add frame %d , %d frames in total'%(i, sample_sum))

        # actually adding the frames once we know the indices
        no_of_frames_added = 0
        for i, frame in enumerate(video):
            if i in indices or i == frame_num-1: # add the last frame always (from original code)
                no_of_frames_added += 1
                framework.add(resize(frame[..., ::-1]))

        if no_of_frames_added != sample_sum+1:
            print(' -- for some reason reason you did not add all the frames picked to be added?')

        print('Computing global features')
        framework.compute()

        end_time = time.monotonic()
        print('Preparations finished in {}!'.format(timedelta(seconds=end_time - start_time)))


    # Main stylization
    video_path_out = os.path.join(result_videos_path, name)
    video_path_out_raw = os.path.join(result_videos_path, name + '_1stPassWithoutAudio.mp4')
    writer = imageio.get_writer(video_path_out_raw, fps=fps)

    # go through the video frames
    start_time = time.monotonic()
    print('Applying style transfer to the video')
    # TODO! update this loop to MoviePy world at some point and not use imageio in the loop
    #  to get rid of the unnecessary disk writes that start to slow down things especially on large clips
    for i, frame in tqdm(enumerate(video)):

        frame = resize(frame[..., ::-1])

        # Crop the image
        H,W,C = frame.shape
        new_input_frame = reshape.process(frame)

        # Stylization
        # TODO! You could redefine the framework for each frame if you are up for glitch for the sake of glitch
        styled_input_frame = framework.transfer(new_input_frame)

        # Crop the image back
        styled_input_frame = styled_input_frame[64:64+H,64:64+W,:]

        # TODO! here you have the stylized frame, "styled_input_frame" and you could add some effects to it
        #  note! that as the whole paper was about temporally coherent style transfer, you could be possibly now
        #  introducing time-varying effects (flickering frames). If this is visually okay for artistic reasons for
        #  you, then this is of course nice

        # cast as unsigned 8-bit integer (not necessarily needed)
        styled_input_frame = styled_input_frame.astype('uint8')

        # add to the output video
        # https://imageio.readthedocs.io/en/stable/examples.html
        writer.append_data(styled_input_frame[..., ::-1])

    writer.close()
    end_time = time.monotonic()
    print('Video style transferred in {}'.format(timedelta(seconds=end_time - start_time)))

    # Now we read the saved .mp4 back to memory and update its audio file
    # TODO! A bit I/O overhead and needless writing, but this repo started with imageio-ffmpeg, and apparently there
    #  is no audio processing? One could do command-line calls for ffmpeg to join the .mp3 as well, but maybe this
    #  Python-only solution is a bit easier?

    # https://stackoverflow.com/a/48866634
    # https://www.programcreek.com/python/example/105718/moviepy.editor.VideoFileClip
    if not audio_track:
        print('Video on the disk is without audio as there was no input audio')
    else:
        print('Reading back the video as the audio was lost, and updating the audio to MP4 with MoviePy')
        start_time = time.monotonic()
        saved_mp4_read_back = mp.VideoFileClip(video_path_out_raw)
        final_clip = saved_mp4_read_back.set_audio(audio_track)
        final_clip.write_videofile(video_path_out, fps=fps, verbose=False, logger=None)
        remove_status = os.remove(video_path_out_raw)
        end_time = time.monotonic()
        print(' TODO! this quick n dirty audio fix added {} seconds of I/O overhead'.format(timedelta(seconds=end_time - start_time)))


    # to write just the audio
    #audio_path_out = video_path_out.replace('mp4', 'mp3')
    #print('writing audio (quick and dirty solution) to disk as separate .mp3 (TODO! combine with the video and get rid of this extra step)')
    #print(' path for audio = {}'.format(audio_path_out))
    #my_clip.audio.write_audiofile(audio_path_out)

    if write_frames_to_disk:

        print('TODO! if your subsequent workflow would prefer PNG frames instead of a video')

        result_frames_path = os.path.join(results_base, 'frames')
        if not os.path.exists(result_frames_path):
            os.mkdir(result_frames_path)

        # Mkdir corresponding folders
        if not os.path.exists('{}/{}'.format(result_frames_path, name)):
            os.mkdir('{}/{}'.format(result_frames_path, name))

    end_time_process = time.monotonic()
    print('Prcessing as a whole took {}'.format(timedelta(seconds=end_time_process - start_time_process)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training of segmentation model for sCROMIS2ICH dataset')
    parser.add_argument('-style_img', '--style_img', type=str, default='../inputs/styles/620ef3a5cd28cd75b742341cc8433eb8.jpg',
                        help='Style img (e.g. jpeg or png)')
    parser.add_argument('-style_img_dir', '--style_img_dir', type=str,
                        default=None, #'../inputs/styles/',
                        help='Directory of all your style images (so you can batch stylize one video with multiple styles and see which you like)')
    parser.add_argument('-input_video', '--input_video', type=str, default=None, #'../inputs/video/scatman.mp4',
                        help='Video input that you want to style (e.g. MP4)')
    parser.add_argument('-input_video_dir', '--input_video_dir', type=str,
                        default=None,
                        help='Directory of all your videos to be stylized (batch processing multiple videos)')
    parser.add_argument('-write_frames_to_disk', '--write_frames_to_disk', type=bool, default=False,
                        help='Writes the frames to disk as well')
    parser.add_argument('-interval', '--interval', type=int, default=8,
                        help="Affects the number of frames needed for 'Sequence-Level Global Feature Sharing', "
                             "i.e. can make your RAM run out if too small (make maybe automatic at point, default was = 8),"
                             "but if you large videos with big resolutions, you might need to ")
    parser.add_argument('-force_on_CPU', '--force_on_CPU', type=bool, default=False,
                        help='Process on CPU instead of GPU')
    parser.add_argument('-not_use_global', '--not_use_global', type=bool, default=True,
                        help='Not using Global Feature Sharing')
    args = parser.parse_args()

    if args.not_use_global:
        use_Global = False

    # Define what video files we are going to process
    if not args.input_video_dir:
        print('Processing a single video file in non-batch mode')
        video_files = [os.path.join(DIR_PATH, args.input_video)]
    else:
        wildcard = '*.*'  # TODO! all supported video formats here, depends on MoviePy, ImageIo, ffmpeg now
        full_video_path = os.path.join(DIR_PATH, args.input_video_dir, wildcard)
        print('Processing all the video files found from {}'.format(full_video_path))
        video_files = sorted(glob.glob(full_video_path))
        print(' found a total of {} video files to be processed'.format(len(video_files)))

    # Define what style images we are going to use for the videos
    if not args.style_img_dir:
        print('Using a single style image for all the videos, or for the single video')
        style_files = [os.path.join(DIR_PATH, args.style_img)]
    else:
        wildcard = '*.*'  # TODO! all supported image formats here, depends on MoviePy, ImageIo, ffmpeg now
        full_style_path = os.path.join(DIR_PATH, args.style_img_dir, wildcard)
        style_files = sorted(glob.glob(full_style_path))
        print('Using all the style images found from {}'.format(args.style_img_dir))
        print(' found a total of {} style images'.format(len(style_files)))

    # PROCESS
    for i, video_file in enumerate(video_files):
        for j, style_file in enumerate(style_files):

            print('\nVideo file #{}/{}, style file #{}/{}'.format(i+1, len(video_files), j+1, len(style_files)))
            process_video(style_img = style_file,
                          input_video = video_file,
                          interval = args.interval,
                          write_frames_to_disk = args.write_frames_to_disk,
                          force_on_CPU = args.force_on_CPU)

    # TODO! you could do a "style_dir" here, so that you process the same video with multiple styles in a loop
    #  i.e. you have a 50 style imgs on a folder and you do not know which works, and you could have the processing
    #  run like overnight, and see the results in the morning

    # TODO! similarly you could have multiple input video(s), and you batch process them with multiple style imgs

    # NOTE! If you are unhappy with the style transfer results and they are not cool enough for you, then you should
    #  try retraining this architecture with your own data, explore other pretrained frameworks, or you can always
    #  try to do something glitchy in the loop / in combination with the style transfer if you are not too picky
    #  about doing style transfer per se, but you just want interesting visuals

    # TODO! You could experiment with denoising, see paper by Lin et al. (2020) on the effects of high spatial frequencies
    #  on style transfer https://arxiv.org/abs/2011.14477 (6.2. The Role of High Frequency Signals)    #
    #  (e.g. old school algorithms BM3D for images, and VBM3D for video, or some recent deep learning based denoising algorithms, see e.g.
    #  https://github.com/wenbihan/reproducible-image-denoising-state-of-the-art
    #  https://github.com/flyywh/Image-Denoising-State-of-the-art
    #  noise2noise -> all its upgrades -> e.g. self2self, noise2void, Neighbor2Neighbor
    #  VIDEO: Self-supervised training for blind multi-frame video denoising / Unsupervised Deep Video Denoising
    #   https://sreyas-mohan.github.io/udvd/ / https://arxiv.org/abs/2011.15045
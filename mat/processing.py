import os
from inference import convert_video, Converter

model_type = 'resnet50' # 'mobilenetv3'
model_checkpoint_path = './checkpoints/rvm_resnet50.pth'
device = 'cuda'

converter = Converter(model_type, model_checkpoint_path, device)

file_name = "tesla"

results_base = './results/'
if not os.path.exists(results_base):
    os.mkdir(results_base)

results_base += file_name + '/'
if not os.path.exists(results_base):
    os.mkdir(results_base)

results_base += model_type + '/'
if not os.path.exists(results_base):
    os.mkdir(results_base)

converter.convert(
    input_source='./inputs/' + file_name + '.mp4',
    input_resize=None,
    output_type='video',
    output_composition=results_base + file_name + '-com.mp4',
    output_alpha=results_base + file_name + '-pha.mp4',
    output_foreground=results_base + file_name + '-fgr.mp4',
    output_video_mbps=4,
    seq_chunk=12,
    num_workers=0
)
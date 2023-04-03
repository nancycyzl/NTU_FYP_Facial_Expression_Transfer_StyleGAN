import os
import cv2
import tqdm
import glob

# mode 1: video to frames
# mode 2: frames to video

mode = 2

if mode == 1:
    video_path = '../data/vid_qx.mp4'
    save_folder = video_path.replace('.mp4', '_frames')
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    count = 0
    while True:
        success,image = vidcap.read()
        if not success:
            break
        cv2.imwrite(os.path.join(save_folder,"frame{:04d}.jpg".format(count)), image)
        count += 1
        print('frame {}/{}'.format(count, total_frames), end='\r')
    print("{} images are extacted in {}.".format(count,save_folder))
    print("video info: fps={}, width={}, height={}".format(fps, int(width), int(height)))


elif mode == 2: 
    frameSize = (1024,1024)
    # frames_folder = 'E:/NTU_FYP/pixel2style2pixel/data/video/vid6_frames_aligned_reconstructed/transfer_result_base_4/alpha0.6/transferred_frames'
    # frames_folder = 'E:/NTU_FYP/pixel2style2pixel/data/video/vid6_frames_aligned_reconstructed/inference_results'
    frames_folder = '../data/vid_qx_results/inference_results'

    out_name = frames_folder + '_compiled.avi'
    out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'DIVX'), 30, frameSize)

    for filename in sorted(os.listdir(frames_folder)):
        file_path = os.path.join(frames_folder, filename)
        img = cv2.imread(file_path)
        out.write(img)

    out.release()

    print('video write to '+ out_name)
import logging
import mimetypes
import os
import time
import torch
import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np

from argparse import ArgumentParser
from tqdm import tqdm
from mmpose.evaluation import CocoMetric
from mmengine.logging import print_log

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def process_one_image(args,
                      img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0,
                      total_frame_count=None,
                      start_time=None):
    """Visualize predicted keypoints (and heatmaps) of one image."""
    # print("Processing one image...")

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # predict keypoints
    # print("Running pose estimation...")
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)

    # Save the visualized results
    if args.show_dir:
        output_file = os.path.join(args.show_dir, os.path.basename(args.input))
        img_vis = visualizer.get_image()
        mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)
        print(f"Saved visualized image to {output_file}")

    return data_samples.get('pred_instances', None)

def process_webcam(args, detector, pose_estimator, visualizer):
    cap = cv2.VideoCapture(0)
    print("Processing webcam frames...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        process_one_image(args, frame, detector, pose_estimator, visualizer)

        if args.show:
            # press ESC to exit
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()

def process_video(args, detector, pose_estimator, visualizer):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {args.input}")
    output_file = None

    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(
            args.output_root, os.path.splitext(os.path.basename(args.input))[0] + '_output.mp4')

    video_writer = None
    pred_instances_list = []
    frame_idx = 0
    frame_count = 0

    print("Processing video frames...")

    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        frame_idx += 1
        frame_count += 1

        if not success:
            break

        # topdown pose estimation
        pred_instances = process_one_image(args, frame, detector,
                                           pose_estimator, visualizer,
                                           0.001, total_frame_count=frame_count, start_time=start_time)

        if args.save_predictions:
            # save prediction results
            video_basename = os.path.splitext(os.path.basename(args.input))[0]
            frame_id_str = f"{frame_idx:05d}"
            image_file_name = f"{video_basename}_{frame_id_str}.jpg"
            pred_instances_list.append(
                dict(
                    image_file=image_file_name,
                    frame_id=frame_idx,
                    instances=split_instances(pred_instances)))

        # output videos
        # if output_file:
        #     frame_vis = visualizer.get_image()

        #     if video_writer is None:
        #         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #         video_writer = cv2.VideoWriter(
        #             output_file,
        #             fourcc,
        #             25,  # saved fps
        #             (frame_vis.shape[1], frame_vis.shape[0]))

        #     video_writer.write(mmcv.rgb2bgr(frame_vis))

        if args.show:
            # press ESC to exit
            if cv2.waitKey(5) & 0xFF == 27:
                break

            time.sleep(args.show_interval)

    if video_writer:
        video_writer.release()

    cap.release()

    # Tính FPS sau khi xử lý toàn bộ video
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = frame_count / elapsed_time
    print(f"Total frames processed: {frame_count}")
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")
    print(f"FPS trung bình của video: {fps:.2f} frames/second")

    if args.save_predictions:
        pred_save_path = os.path.join(
            args.output_root, os.path.splitext(os.path.basename(args.input))[0] + '_results.json')
        with open(pred_save_path, 'w') as f:
            json.dump(
                dict(
                    meta_info=pose_estimator.dataset_meta,
                    instance_info=pred_instances_list),
                f,
                indent='\t')
        print(f'Predictions have been saved at {pred_save_path}')

    if output_file:
        print_log(
            f'The output video has been saved at {output_file}',
            logger='current',
            level=logging.INFO)

    return frame_count  # Trả về số khung hình đã xử lý


def process_image_folder(args, detector, pose_estimator, visualizer):
    """Process a folder containing multiple subfolders, each with images."""
    input_root = args.input_folder_dir
    output_root = args.output_root
    save_predictions = args.save_predictions

    missing_kpts_images = []  # Danh sách ảnh thiếu keypoints

    for subdir in tqdm(os.listdir(input_root), desc="Processing subfolders"):
        subdir_path = os.path.join(input_root, subdir)
        if not os.path.isdir(subdir_path):
            continue
        output_subdir = os.path.join(output_root, subdir)
        os.makedirs(output_subdir, exist_ok=True)
        image_files = [
            os.path.join(subdir_path, f)
            for f in os.listdir(subdir_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
        pred_instances_all = []
        for image_file in tqdm(image_files, desc=f"Processing images in {subdir}", leave=False):
            args.input = image_file
            pred_instances = process_one_image(args, image_file, detector, pose_estimator, visualizer)
            # Kiểm tra missing keypoints
            is_missing = False
            if pred_instances is not None and len(pred_instances) > 0:
                for inst in pred_instances:
                    kpts = inst.keypoints if hasattr(inst, 'keypoints') else inst['keypoints']
                    # Nếu tất cả kepoints đều bằng 0 hoặc số lượng < 17
                    if (np.sum(np.array(kpts)[:, :2]) == 0) or (len(kpts) < 17):
                        is_missing = True
                        break
            else:
                is_missing = True
            if is_missing:
                missing_kpts_images.append(image_file)

            if save_predictions:
                pred_instances_all.append(
                    dict(
                        image_file=os.path.basename(image_file),
                        instances=split_instances(pred_instances))
                )
            # Save visualized image
            if args.output_root:
                img_vis = visualizer.get_image()
                output_file = os.path.join(output_subdir, os.path.basename(image_file))
                mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)
                
        # Lưu file json riêng cho từng subfolder
        if save_predictions:
            pred_save_path = os.path.join(output_subdir, f'{subdir}_results.json')
            with open(pred_save_path, 'w') as f:
                json.dump(
                    dict(
                        meta_info=pose_estimator.dataset_meta,
                        instance_info=pred_instances_all),
                    f,
                    indent='\t')
            print(f'Predictions for {subdir} have been saved at {pred_save_path}')

    # Sau khi chạy xong, lưu danh sách ảnh missing kpts ra file và đếm số lượng
    if missing_kpts_images:
        missing_file = os.path.join(output_root, 'missing_kpts_images.txt')
        with open(missing_file, 'w') as f:
            for img_path in missing_kpts_images:
                f.write(img_path + '\n')
        print(f"Saved missing keypoints image list to {missing_file}")
        print(f"Total missing keypoints images: {len(missing_kpts_images)}")

def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """

    ### BẮT ĐẦU SỬA ĐỔI 1: THÊM CẤU HÌNH STREAM ###
    # --- Cấu hình cho việc stream camera từ Windows sang WSL ---
    # !!! QUAN TRỌNG: Hãy đảm bảo địa chỉ IP này khớp với địa chỉ IP của máy Windows của bạn !!!
    WINDOWS_HOST_IP = "192.168.1.18" 
    STREAM_PORT = 5000
    WEBCAM_STREAM_URL = f"http://{WINDOWS_HOST_IP}:{STREAM_PORT}/video_feed"
    # --- Kết thúc cấu hình ---
    ### KẾT THÚC SỬA ĐỔI 1 ###


    print("Starting inference...")  # Add this to notify the start of inference process
    input_type = None

    parser = ArgumentParser()
    parser.add_argument('det_config',nargs= '?', default='/home/phmlog/Document/mmpose/projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py', help='Config file for detection')
    parser.add_argument('det_checkpoint',nargs= '?', default='/home/phmlog/Document/mmpose/models/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth', help='Checkpoint file for detection')
    parser.add_argument('pose_config',nargs= '?', default='/home/phmlog/Document/mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-l_8xb256-420e_coco-256x192.py', help='Config file for pose')
    parser.add_argument('pose_checkpoint',nargs= '?', default='/home/phmlog/Document/mmpose/models/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth', help='Checkpoint file for pose')
    parser.add_argument(
        '--input', type=str, default='', help='Image/Video file')
    parser.add_argument(
        '--input-dir',
        type=str,
        default='',
        help='Directory containing multiple video files to process.')
    parser.add_argument(
        '--input-img-dir',
        type=str,
        default=None,
        help='Directory containing multiple image files to process.')
    parser.add_argument(
        '--input-folder-dir',
        type=str,
        default=None,
        help='Root folder containing multiple subfolders of images to process.')
    parser.add_argument(
        '--img-size',
        type=int,
        default=960,
        help='Resize input image to this size before inference (square resize)')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--output-root',
        type=str,
        default='/home/phmlog/Document/open-mmlab/mmpose/output_',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='whether to save predicted results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--draw-bbox', action='store_true', help='Draw bboxes of instances')
    parser.add_argument(
        '--show-dir',  # Thêm tham số show-dir
        type=str,
        help='Directory to save visualized images.'
    )

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    if 'cuda' in args.device and torch.cuda.is_available():
        print(f"Using device: {args.device}")
    else:
        print("Using CPU for inference.")

    # assert args.show or (args.output_root != '')
    # assert args.show or (args.show_dir != '')

    # assert args.input != ''
    # assert args.input_dir != ''
    assert args.input_img_dir != ''
    assert args.input_folder_dir != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None
    

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root,
                                   os.path.basename(args.input))
        if args.input == 'webcam':
            output_file += '.mp4'

    if args.save_predictions:
        assert args.output_root != ''
        # assert args.show_dir != ''
        args.pred_save_path = f'{args.output_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}.json'

    print("Initializing detection model...")
    # build detector
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    print("Initializing pose estimator...")
    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    print("Building visualizer...")
    # build visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

    print("Starting to process input...")

    if args.input_dir:
        # Lấy danh sách tất cả các tệp video trong thư mục
        video_files = [
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ]

        if not video_files:
            raise ValueError(f"No video files found in directory: {args.input_dir}")

        print(f"Found {len(video_files)} video(s) in directory: {args.input_dir}")

        total_frame_count = 0
        total_elapsed_time = 0

        for video_file in tqdm(video_files, desc="Processing videos"):
            print(f"Processing video: {video_file}")
            args.input = video_file  # Cập nhật `args.input` để xử lý từng video

            # Bắt đầu tính thời gian cho video này
            start_time = time.time()

            # Gọi hàm xử lý video
            frame_count = process_video(args, detector, pose_estimator, visualizer)

            # Kết thúc tính thời gian cho video này
            end_time = time.time()
            elapsed_time = end_time - start_time

            # Cập nhật tổng số khung hình và thời gian
            total_frame_count += frame_count  # frame_count được tính trong process_video
            total_elapsed_time += elapsed_time

            if total_elapsed_time > 0:
                # Tính FPS tổng hợp sau khi xử lý tất cả các video
                total_fps = total_frame_count / total_elapsed_time
            else:
                total_fps = 0
        # else:
        #     if args.input == 'webcam':
        #         input_type = 'webcam'
        #     else:
        #         input_type = mimetypes.guess_type(args.input)[0].split('/')[0]

        print(f"input_type: {input_type}")  # Thêm dòng này để kiểm tra input_type

        # In ra thông tin tổng hợp
        print(f"Total frames processed across all videos: {total_frame_count}")
        print(f"Total elapsed time across all videos: {total_elapsed_time:.2f} seconds")
        print(f"Total FPS across all videos: {total_fps:.2f} frames/second")
        return
    
    if args.input_img_dir:
        image_files = [
            os.path.join(args.input_img_dir, f)
            for f in os.listdir(args.input_img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]

        if not image_files:
            raise ValueError(f"No image files found in directory: {args.input_img_dir}")

        print(f"Found {len(image_files)} image(s) in directory: {args.input_img_dir}")

        # Tạo thư mục output nếu chưa có
        if args.output_root:
            mmengine.mkdir_or_exist(args.output_root)

        pred_instances_all = []

        start_time = time.time()

        for image_file in tqdm(image_files, desc="Processing images"):
            print(f"Processing image: {image_file}")
            args.input = image_file  # Cập nhật `args.input` để xử lý từng ảnh

            pred_instances = process_one_image(args, image_file, detector, pose_estimator, visualizer)

            if args.save_predictions:
                # pred_instances_list = split_instances(pred_instances)
                # Lưu kết quả dự đoán cho từng ảnh nếu muốn
                pred_instances_all.append(
                    dict(
                        image_file=os.path.basename(image_file),
                        instances=split_instances(pred_instances))
                    )


            # Lưu ảnh đã inference vào output_root
            if args.output_root:
                img_vis = visualizer.get_image()
                output_file = os.path.join(args.output_root, os.path.basename(image_file))
                mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)
                print(f"Saved visualized image to {output_file}")

        # Kết thúc tính thời gian
        elapsed_time = time.time() - start_time
        fps = len(image_files) / elapsed_time
        print(f"Total images processed: {len(image_files)}")
        print(f"Total elapsed time: {elapsed_time:.2f} seconds")
        print(f"FPS trên thư mục ảnh: {fps:.2f} frames/second")

                
        # Lưu tất cả kết quả keypoints ra file JSON sau khi xử lý xong
        if args.save_predictions:
            pred_save_path = os.path.join(
                args.output_root, 'all_images_results.json')
            with open(pred_save_path, 'w') as f:
                json.dump(
                    dict(
                        meta_info=pose_estimator.dataset_meta,
                        instance_info=pred_instances_all),
                    f,
                    indent='\t')
            print(f'Predictions for all images have been saved at {pred_save_path}')

        return     

    # Thêm xử lý cho input_folder_dir
    if args.input_folder_dir:
        if not os.path.isdir(args.input_folder_dir):
            raise ValueError(f"Input folder dir not found: {args.input_folder_dir}")
        if args.output_root:
            mmengine.mkdir_or_exist(args.output_root)
        process_image_folder(args, detector, pose_estimator, visualizer)
        return
   
    
    # ĐẶT ĐOẠN NÀY TRƯỚC print(f"input_type: {input_type}")
    if args.input == 'webcam':
        input_type = 'webcam'
    elif args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        input_type = 'video'
    elif args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        input_type = 'image'
    else:
        input_type = None

    print(f"input_type: {input_type}")  # Thêm dòng này để kiểm tra input_type


    if input_type == 'image':
        print("Processing input image...")
        # inference
        pred_instances = process_one_image(args, args.input, detector,
                                           pose_estimator, visualizer)

        if args.save_predictions:
            pred_instances_list = split_instances(pred_instances)

        if output_file:
            img_vis = visualizer.get_image()
            mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)

    elif input_type in ['webcam', 'video']:
        ### BẮT ĐẦU SỬA ĐỔI 2: THAY ĐỔI NGUỒN VIDEO CAPTURE ###
        if args.input == 'webcam':
            print(f"Connecting to Windows webcam stream at: {WEBCAM_STREAM_URL}")
            cap = cv2.VideoCapture(WEBCAM_STREAM_URL)  # <--- ĐÂY LÀ THAY ĐỔI QUAN TRỌNG
            total_frames = None
        else: # input_type == 'video'
            cap = cv2.VideoCapture(args.input)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ### KẾT THÚC SỬA ĐỔI 2 ###

        if not cap.isOpened():
            print(f"Lỗi: Không thể mở video stream từ: {args.input}")
            if args.input == 'webcam':
                print("- Server trên Windows (stream_server.py) đã chạy chưa?")
                print("- Địa chỉ IP và Port trong script này có đúng không?")
                print("- Windows Firewall đã cho phép kết nối chưa?")
            return

        video_writer = None
        pred_instances_list = []
        frame_count = 0
        frame_idx = 0

        print("Processing video frames...")

        start_time = time.time() 

        # Sử dụng tqdm cho progress bar nếu là video file
        frame_iter = range(total_frames) if total_frames is not None else iter(int, 1)
        for _ in tqdm(frame_iter, desc="Processing frames", unit="frame"):
            success, frame = cap.read()
            if not success:
                break
            frame_idx += 1
            frame_count += 1

            # topdown pose estimation
            pred_instances = process_one_image(args, frame, detector,
                                               pose_estimator, visualizer,
                                               0.001, total_frame_count=frame_count, start_time=start_time)

            if args.save_predictions:
                # save prediction results
                pred_instances_list.append(
                    dict(
                        frame_id=frame_idx,
                        instances=split_instances(pred_instances)))

            # output videos
            if output_file:
                frame_vis = visualizer.get_image()

                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # the size of the image with visualization may vary
                    # depending on the presence of heatmaps
                    video_writer = cv2.VideoWriter(
                        output_file,
                        fourcc,
                        25,  # saved fps
                        (frame_vis.shape[1], frame_vis.shape[0]))

                video_writer.write(mmcv.rgb2bgr(frame_vis))

            if args.show:
                # press ESC to exit
                if cv2.waitKey(5) & 0xFF == 27:
                    break

                time.sleep(args.show_interval)

        if video_writer:
            video_writer.release()

        cap.release()
        cv2.destroyAllWindows() # Thêm dòng này để đóng cửa sổ khi kết thúc


        # Tính FPS sau khi xử lý toàn bộ video
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = frame_count / elapsed_time
        print(f"Total frames processed: {frame_count}")
        print(f"Total elapsed time: {elapsed_time:.2f} seconds")
        print(f"FPS trung bình của video: {fps:.2f} frames/second")

    else:
        args.save_predictions = False
        raise ValueError(
            f'file {os.path.basename(args.input)} has invalid format.')

    if args.save_predictions:
        with open(args.pred_save_path, 'w') as f:
            json.dump(
                dict(
                    meta_info=pose_estimator.dataset_meta,
                    instance_info=pred_instances_list),
                f,
                indent='\t')
        print(f'predictions have been saved at {args.pred_save_path}')

    if output_file:
        input_type = input_type.replace('webcam', 'video')
        print_log(
            f'the output {input_type} has been saved at {output_file}',
            logger='current',
            level=logging.INFO)



if __name__ == '__main__':
    main()




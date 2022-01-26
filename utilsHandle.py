import os
import sys
from pathlib import Path
import cv2
import torch
from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import (LOGGER, check_img_size, colorstr,
                           increment_path, non_max_suppression, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from os.path import join
from os import listdir
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image

dir_path = os.path.dirname(os.path.realpath(__file__))
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

config = Cfg.load_config_from_name('vgg_transformer')

config['weights'] = 'transformerocr.pth'
config['cnn']['pretrained'] = False
config['device'] = 'cuda:0'
config['predictor']['beamsearch'] = False
detector = Predictor(config)


class pathImg:
    def __init__(self):
        path_card = ''
        path_avatar = ''

    def set_path_card(self, path):
        self.path_Card = path

    def set_path_avatar(self, path):
        self.path_avatar = path


pathImg = pathImg()


def get_path():
    return pathImg


def predict(source):
    cropImg(source)
    list_Dirs = [f for f in listdir(join(dir_path, 'runs/detect'))]
    full_path = join(join(dir_path, 'runs/detect'), join(list_Dirs[-1], 'crops'))
    list_cropped = [f for f in listdir(full_path)]
    result = {}
    for folderPath in list_cropped:
        print(folderPath)
        path_folder = join(full_path, folderPath)
        path_imgs = [f for f in listdir(path_folder)]
        count = 0
        for imgPath in path_imgs:
            img = Image.open(join(path_folder, imgPath))
            if folderPath not in ['avatar', 'card']:
                s = detector.predict(img)
                print(s)
                split_res = s.split(":")
                if len(split_res) == 2:
                    result[split_res[0].strip()] = split_res[1].strip()
                    continue
                if count == 0:
                    result[folderPath] = s
                else:
                    result[folderPath + str(count)] = s
                count += 1
            elif folderPath == 'avatar':
                pathImg.set_path_avatar(join(path_folder, imgPath))
            elif folderPath == 'card':
                pathImg.set_path_card(join(path_folder, imgPath))
    print(result)
    return result


def cropImg(source):
    save_txt = False
    imgsz = (640, 640)

    weights = 'models/best_ver3.pt'
    data = ROOT / 'data/coco128.yaml'

    nosave = False

    conf_thres = 0.25  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 1000  # maximum detections per image

    view_img = False  # show results

    save_conf = False  # save confidences in --save-txt labels
    save_crop = True  # save cropped prediction boxes

    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False  # class-agnostic NMS
    augment = False  # augmented inference
    visualize = False  # visualize features
    update = False  # update all models
    project = ROOT / 'runs/detect'  # save results to project/name
    line_thickness = 3  # bounding box thickness (pixels)
    hide_labels = False  # hide labels
    hide_conf = False  # hide confidences
    half = False  # use FP16 half-precision inference

    save_img = not nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = increment_path(Path(project) / 'exp', exist_ok=False)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device('')
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

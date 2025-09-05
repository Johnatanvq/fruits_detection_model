import os, sys, glob, time, argparse
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO
import depthai as dai

def parse_args():
    """Parse CLI arguments for model path, input source, thresholds and runtime options."""
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Path to YOLO weights (e.g., best.pt)')
    p.add_argument('--source', required=True, help='img | dir | video.mp4 | usb0 (OAK-1)')
    p.add_argument('--thresh', type=float, default=0.5, help='Minimum confidence to draw')
    p.add_argument('--resolution', default='640x480', help='Output/display resolution WxH')
    p.add_argument('--record', action='store_true', help='Record output video (demo1.avi)')
    return p.parse_args()

def open_oak(resW, resH):
    """
    Initialize a simple DepthAI pipeline to stream RGB preview frames
    from an OAK-1 device at the requested resolution.
    """
    pipe = dai.Pipeline()
    cam = pipe.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setFps(30)
    cam.setPreviewSize(resW, resH)                  # preview stream size
    cam.setPreviewKeepAspectRatio(False)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    xout = pipe.create(dai.node.XLinkOut)
    xout.setStreamName('rgb')
    cam.preview.link(xout.input)

    dev = dai.Device(pipe)
    q = dev.getOutputQueue('rgb', maxSize=4, blocking=False)
    return dev, q

def frame_iter(source, resW, resH):
    """
    Unified frame generator:
      - Directory: iterate all images
      - Single image: yield once
      - Video file: iterate frames
      - 'usb*' (e.g., 'usb0'): stream from OAK-1
    Frames are resized to (resW, resH) for consistent inference/display.
    """
    img_ext = {'.jpg','.jpeg','.png','.bmp','.JPG','.JPEG','.PNG','.BMP'}
    vid_ext = {'.avi','.mov','.mp4','.mkv','.wmv'}
    p = Path(source)

    if p.is_dir():
        # Iterate images in directory (sorted for reproducibility)
        for f in sorted(glob.glob(str(p/'*'))):
            if Path(f).suffix in img_ext:
                yield cv2.resize(cv2.imread(f), (resW, resH))

    elif p.is_file():
        # Single file: image or video
        if p.suffix in img_ext:
            yield cv2.resize(cv2.imread(str(p)), (resW, resH))
        elif p.suffix in vid_ext:
            cap = cv2.VideoCapture(str(p))
            while True:
                ok, fr = cap.read()
                if not ok: break
                yield cv2.resize(fr, (resW, resH))
            cap.release()

    elif source.startswith('usb'):
        # OAK-1 live stream via DepthAI
        dev, q = open_oak(resW, resH)
        try:
            while True:
                m = q.tryGet()
                if m is None:
                    # allow UI to process events / ESC to exit
                    if cv2.waitKey(1) == 27: break
                    continue
                yield m.getCvFrame()
        finally:
            dev.close()
    else:
        print('Invalid source; expected image/dir/video or usb* (OAK-1).')
        sys.exit(1)

def draw_dets(im, boxes, names, thresh):
    """
    Draw bounding boxes and labels for detections >= thresh.
    Returns the number of drawn detections.
    """
    colors = [
        (164,120,87),(68,148,228),(93,97,209),(178,182,133),(88,159,106),
        (96,202,231),(159,124,168),(169,162,241),(98,118,150),(172,176,184)
    ]
    n = 0
    for b in boxes:
        conf = float(b.conf.item())
        if conf < thresh:
            continue
        # xyxy in absolute pixel coords
        x1,y1,x2,y2 = b.xyxy.cpu().numpy().astype(int).squeeze()
        cls = int(b.cls.item())
        color = colors[cls % len(colors)]
        cv2.rectangle(im, (x1,y1), (x2,y2), color, 2)
        cv2.putText(
            im,
            f'{names.get(cls, str(cls))}:{int(conf*100)}%',
            (x1, max(20, y1-5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
        n += 1
    return n

def main():
    """Main inference loop: load model, iterate frames, run YOLO, draw, show/record."""
    a = parse_args()

    # Ensure model path exists; YOLO() will load a.pt file from this path.
    if not Path(a.model).exists():
        print('Model not found'); sys.exit(1)

    # Parse resolution "WxH" (e.g., "640x480")
    resW, resH = map(int, a.resolution.lower().split('x'))

    # Load YOLO model for detection task (consumes the file passed in --model)
    model = YOLO(a.model, task='detect')
    names = model.names

    # Optional recorder (MJPG @ 30 FPS)
    writer = None
    if a.record:
        writer = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))

    # Simple FPS tracker over last ~200 frames
    fps_hist, t0 = [], time.perf_counter()

    # Main frame loop
    for fr in frame_iter(a.source, resW, resH):
        t1 = time.perf_counter()

        # Run inference; r is a Results object, take first batch element
        r = model(fr, verbose=False)[0]

        # Draw detections above threshold
        count = draw_dets(fr, r.boxes, names, a.thresh)

        # Compute instantaneous FPS and show rolling average
        fps = 1.0 / max(1e-6, time.perf_counter() - t1)
        fps_hist.append(fps); fps_hist = fps_hist[-200:]
        cv2.putText(
            fr,
            f'FPS:{np.mean(fps_hist):.2f}  Obj:{count}',
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2
        )

        # Display and optional recording
        cv2.imshow('Detection', fr)
        if writer:
            writer.write(fr)

        # Keyboard: 'q' or ESC to exit; non-blocking for streams, blocking for stills
        k = cv2.waitKey(1 if a.source.startswith(('usb',)) or Path(a.source).suffix else 0)
        if k in (ord('q'), 27):
            break

    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

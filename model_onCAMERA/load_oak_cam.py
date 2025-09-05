import cv2
import depthai as dai
import time

# ---------------------------------------------------------------------
# Model & inference configuration
# ---------------------------------------------------------------------
# Path to the compiled blob for OAK-1 (DepthAI). Keep the blob outside
# the repo (e.g., Hugging Face / Releases) and point this path to the
# local copy downloaded by your script or README instructions.
model_blob_path = "my_model_openvino_2022.1_6shave.blob"

# Basic runtime settings for the on-device detector.
# NOTE: Ensure these thresholds align with your training/evaluation criteria.
config = {
    "input_size": 640,                 # Network input (square, WxH)
    "confidence_threshold": 0.5,       # Min confidence to report a detection
    "iou_threshold": 0.5,              # IoU threshold for NMS inside Yolo NN
    "labels": ["apple", "carrot", "orange"]  # Class names in the exact training order
}

# Per-class BGR colors for drawing (only affects visualization).
class_colors = {
    "apple":  (0,   0, 255),  # Red
    "carrot": (0, 255,   0),  # Green
    "orange": (0, 165, 255),  # Orange
}

# ---------------------------------------------------------------------
# DepthAI pipeline definition
# ---------------------------------------------------------------------
pipeline = dai.Pipeline()

# Nodes:
# - ColorCamera: produces a preview stream at the desired network size
# - YoloDetectionNetwork: runs the YOLO model on-device
# - XLinkOut (rgb): sends passthrough RGB frames to host for display
# - XLinkOut (detections): sends detection results to host
cam_rgb       = pipeline.create(dai.node.ColorCamera)
detection_nn  = pipeline.create(dai.node.YoloDetectionNetwork)
xout_rgb      = pipeline.create(dai.node.XLinkOut)
xout_nn       = pipeline.create(dai.node.XLinkOut)

# ---------------------------------------------------------------------
# Camera configuration
# ---------------------------------------------------------------------
# Preview size MUST match the network input expected by your blob.
# The sensor resolution can be higher; the preview is cropped/resized
# internally to feed the NN. Keep aspect ratio considerations in mind.
cam_rgb.setPreviewSize(config["input_size"], config["input_size"])
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam_rgb.setFps(30)

# ---------------------------------------------------------------------
# YOLO network configuration
# ---------------------------------------------------------------------
# Load the compiled blob and set inference thresholds and model specifics.
detection_nn.setBlobPath(model_blob_path)
detection_nn.setConfidenceThreshold(config["confidence_threshold"])
detection_nn.setNumClasses(3)     # Number of classes must match your training
detection_nn.setCoordinateSize(4) # xywh/xyxy per detection (DepthAI expects 4)
detection_nn.setIouThreshold(config["iou_threshold"])
detection_nn.input.setBlocking(False)

# IMPORTANT:
# Anchors and masks are MODEL-SPECIFIC. They must match the training/export
# configuration used to produce your blob. If they don't match, detection
# quality will degrade or fail. Adjust as needed for your model version.
# For YOLOv5/6/11 on DepthAI via YoloDetectionNetwork, use the anchors/masks
# that were used during training (or that your exporter reports).
detection_nn.setAnchors([
    10, 13, 16, 30, 33, 23,
    30, 61, 62, 45, 59, 119,
    116, 90, 156, 198, 373, 326
])
# NOTE: The mask names (e.g., "side26", "side13") are tied to model strides/scales.
# Ensure they correspond to the scales your exporter reports for your blob.
detection_nn.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})

# Link camera → NN, and create two output streams: RGB passthrough and detections
cam_rgb.preview.link(detection_nn.input)
detection_nn.passthrough.link(xout_rgb.input)
detection_nn.out.link(xout_nn.input)

# Name the output streams for host-side queues
xout_rgb.setStreamName("rgb")
xout_nn.setStreamName("detections")

# ---------------------------------------------------------------------
# Host-side execution
# ---------------------------------------------------------------------
# Open the device and run the pipeline. Using a context manager ensures
# proper release of resources even if an exception occurs.
with dai.Device(pipeline) as device:
    print("Device connected, MXID:", device.getMxId())

    # Non-blocking output queues let the UI remain responsive.
    q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    q_det = device.getOutputQueue("detections", maxSize=4, blocking=False)

    # Simple FPS computation over a sliding window of 10 frames
    start_time = time.monotonic()
    counter = 0
    fps = 0.0

    while True:
        # Retrieve the latest frames/results. get() blocks until a message arrives.
        in_rgb = q_rgb.get()
        in_det = q_det.get()

        # Convert NN passthrough to OpenCV Mat (BGR)
        frame = in_rgb.getCvFrame()

        # Update FPS every 10 frames to avoid excessive fluctuations
        counter += 1
        if counter % 10 == 0:
            fps = 10 / (time.monotonic() - start_time)
            start_time = time.monotonic()

        # Overlay FPS text
        cv2.putText(
            frame, f"FPS: {fps:.2f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        # -----------------------------------------------------------------
        # Post-processing: draw YOLO detections on the RGB frame
        # -----------------------------------------------------------------
        # Each detection contains normalized [0..1] coords and metadata.
        detections = in_det.detections
        for detection in detections:
            # Denormalize coordinates to pixel space
            x1 = int(detection.xmin * frame.shape[1])
            y1 = int(detection.ymin * frame.shape[0])
            x2 = int(detection.xmax * frame.shape[1])
            y2 = int(detection.ymax * frame.shape[0])

            # Map class id → human-readable label and pick a draw color
            label = config["labels"][detection.label]
            confidence = detection.confidence
            color = class_colors[label]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw a filled label background for readability, then print text
            label_text = f"{label} {confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(
                frame, label_text, (x1, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
            )

        # Show the annotated frame
        cv2.imshow("YOLOv11 on OAK-1", frame)

        # Quit on 'q' or ESC
        key = cv2.waitKey(1)
        if key in (ord('q'), 27):
            break

# Clean up any remaining OpenCV windows
cv2.destroyAllWindows()

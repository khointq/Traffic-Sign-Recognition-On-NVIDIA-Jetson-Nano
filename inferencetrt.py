import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import os
import cv2 
from typing import List, Tuple
from playsound import playsound
import time
import threading

trt_logger = trt.Logger(trt.Logger.VERBOSE)

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def load_engine(engine_file_path):
  assert os.path.exists(engine_file_path)
  print("Reading engine from file {}".format(engine_file_path))
  trt.init_libnvinfer_plugins(trt_logger, "")
  with open(engine_file_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
    serialized_engine = f.read()
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    print(type(engine))  
    return engine

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)[1:]) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        binding_index = engine.get_binding_index(binding)

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        # Only bytes, no need for size
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))

        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

class InferenceSession:
  def __init__(self, engine_file, inference_shape: Tuple[int,int]):
    self.engine = load_engine(engine_file)
    self.context = None
    self.inference_shape = inference_shape

  def __enter__(self):
      self.context = self.engine.create_execution_context()
      assert self.context

      self.context.set_input_shape('input', (1, 3, *self.inference_shape))
      self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine,self.context)

      return self

  def preprocess(self, image):
      image = np.array(image)
      rows, cols = self.inference_shape
      original_shape = image.shape[:2]
      # Resize image to fixed size
      image = cv2.resize(np.array(image), dsize=(cols, rows))
      image=image[:,:,::-1]
      # Switch from HWC to to CHW order
      return np.moveaxis(image, -1, 0), original_shape

  def postprocess(self, detected_boxes, original_shape: Tuple[int, int]):
      sx = original_shape[1] / self.inference_shape[1]
      sy = original_shape[0] / self.inference_shape[0]
      detected_boxes[:, :, [0, 2]] *= sx
      detected_boxes[:, :, [1, 3]] *= sy
      return detected_boxes

  def __call__(self, image):
      batch_size = 1
      input_image, original_shape = self.preprocess(image)

      self.inputs[0].host[:np.prod(input_image.shape)] = np.asarray(input_image).ravel()

      [cuda.memcpy_htod(inp.device, inp.host) for inp in self.inputs]
      success = self.context.execute_v2(bindings=self.bindings)
      assert success
      [cuda.memcpy_dtoh(out.host, out.device) for out in self.outputs]

      num_detections, detected_boxes, detected_scores, detected_labels = [o.host for o in self.outputs]

      num_detections = num_detections.reshape(-1)
      num_predictions_per_image = len(detected_scores) // batch_size
      detected_boxes  = detected_boxes.reshape(batch_size, num_predictions_per_image, 4)
      detected_scores = detected_scores.reshape(batch_size, num_predictions_per_image)
      detected_labels = detected_labels.reshape(batch_size, num_predictions_per_image)

      detected_boxes = self.postprocess(detected_boxes, original_shape) # Scale coordinates back to original image shape
      return num_detections, detected_boxes, detected_scores, detected_labels


  def __exit__(self, exc_type, exc_val, exc_tb):
    del self.inputs, self.outputs, self.bindings, self.stream, self.context

VIDEO_PATH = "video_demo" 
MODEL_PATH = "model.trt"
CLASS_NAMES=['cam_re_trai',
 'cam_re_phai',
 'toc_do_60',
 'toc_do_30',
 'toc_do_40',
 'toc_do_35',
 'toc_do_100',
 'toc_do_80',
 'cam_dung',
 'toc_do_50',
 'cam_oto',
 'nguoi_di_bo_cat_ngang',
 'cam_quay_dau',
 'duong_cua',
 'tron_truot',
 'den_giao_thong',
 'duong_hep',
 'nguy_hiem',
 'cam_vuot',
 'het_khu_dan_cu',
 'het_cam_vuot',
 'cam_re_phai_quay_dau',
 'duong_gho_ghe',
 'di_cham',
 'cong_truong',
 'duong_hai_chieu',
 'cam_re_trai_quay_dau',
 'toc_do_70',
 'cam_re_phai_di_thang',
 'cam_re_hai_ben',
 'cam_re_trai_di_thang',
 'xuong_doc',
 'duong_doc',
 'tau_lua',
 'khu_dan_cu',
 'vuc_sau',
 'toc_do_120',
 'toc_do_90',
 'dung_lai',
 'cam_ken',
 'tai_nan']

SPEED_CLASSES = ['toc_do_30', 'toc_do_40', 'toc_do_50', 'toc_do_60']
AUDIO_DIR = "AUDIO"

SPEED_AUDIO = {
    cls: os.path.join(AUDIO_DIR, f"{cls}.mp3")
    for cls in SPEED_CLASSES
}
def play_sound(path):
    if os.path.exists(path):
        playsound(path)
    else:
        print(f"[Warn] Audio file not found: {path}")

cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
last_play_times = {cls: 0 for cls in SPEED_CLASSES}
COOLDOWN = 5  

with InferenceSession(MODEL_PATH, inference_shape=(640, 640)) as session:

    cuda.Context.synchronize()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # === Inference ===
        preds = session.run(frame)
        num_dets, boxes, scores, labels = preds
        current_speeds = set()
        num = int(num_dets[0])
        for i in range(num):
            x1, y1, x2, y2 = boxes[0][i]
            score = scores[0][i]
            cls_id = int(labels[0][i])

            # Scale box từ 640x640 về frame gốc
            scale_x = width / 640
            scale_y = height / 640
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            

            if label in SPEED_CLASSES and score>=0.9:
                current_speeds.add(label)

        # Với mỗi speed class đang detect, check cooldown rồi play
        now = time.time()
        for cls in current_speeds:
            if now - last_play_times[cls] >= COOLDOWN:
                audio_path = SPEED_AUDIO[cls]
                threading.Thread(target=play_sound, args=(audio_path,), daemon=True).start()
                last_play_times[cls] = now

        cv2.imshow("YOLO-NAS Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
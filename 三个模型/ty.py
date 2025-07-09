import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageFont, ImageDraw
import cv2
import numpy as np
import threading
import os
from datetime import datetime
import queue
import time
import subprocess
from rknnlite.api import RKNNLite

# 类别名
class_names = ['佩戴头盔', '未佩戴头盔', '穿反光衣', '未穿反光衣']

# 全局变量
model = None
stop_flag = False
video_file_path = ""
last_alert_time = {}

frame_queue = queue.Queue(maxsize=10)
result_queue = queue.Queue(maxsize=10)
frame_skip = 0

video_capture = None
video_thread = None
inference_thread = None
video_running = False

# 帧率计算
fps_counter = 0
last_fps_time = time.time()
current_fps = 0

# 颜色定义
BG_COLOR = "#2c3e50"
BUTTON_COLOR = "#3498db"
TEXT_COLOR = "#ecf0f1"
LOG_BG = "#34495e"
LOG_FG = "#ecf0f1"
PANEL_BG = "#2c3e50"

# 字体定义
FONT_TITLE = ("微软雅黑", 12, "bold")
FONT_BUTTON = ("微软雅黑", 10)
FONT_LOG = ("Consolas", 9)

# 语音播报管理
voice_process = None
speech_lock = threading.Lock()
no_helmet_flag = False

def log(msg):
    """记录日志"""
    print(f"[LOG] {msg}")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_text.insert(tk.END, f"[{now}] {msg}\n")
    log_text.see(tk.END)
    log_text.update()

def update_fps():
    """更新帧率显示"""
    global fps_counter, last_fps_time, current_fps
    fps_counter += 1
    current_time = time.time()
    if current_time - last_fps_time >= 1.0:
        current_fps = fps_counter / (current_time - last_fps_time)
        fps_counter = 0
        last_fps_time = current_time
        fps_label.config(text=f"帧率: {current_fps:.1f} FPS")
    window.after(100, update_fps)

def xywh2xyxy(x):
    """将xywh格式转换为xyxy格式"""
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def filter_boxes(boxes, scores, classes, score_threshold=0.25):
    """根据置信度过滤检测框"""
    mask = scores >= score_threshold
    return boxes[mask], classes[mask], scores[mask]

def nms_boxes(boxes, scores, iou_threshold=0.45):
    """非极大值抑制"""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep)

def yolov5_post_process(output, conf_thres=0.25, iou_thres=0.45):
    """YOLOv5后处理"""
    # 输出形状: [1, 25200, 8] (batch, num_anchors, 4+1+3)
    if len(output.shape) == 3:
        output = output.squeeze(0)  # [25200, 8]
    
    # 分离box、confidence和class
    boxes = output[:, :4]  # x,y,w,h
    conf = output[:, 4:5]  # confidence
    cls_probs = output[:, 5:]  # class probabilities
    
    # 计算每个框的类别分数
    scores = conf * cls_probs
    class_ids = np.argmax(scores, axis=1)
    class_scores = np.max(scores, axis=1)
    
    # 过滤低分检测
    boxes, class_ids, class_scores = filter_boxes(boxes, class_scores, class_ids, conf_thres)
    if boxes.shape[0] == 0:
        return None, None, None
    
    # 转换坐标格式
    boxes = xywh2xyxy(boxes)
    
    # NMS处理
    keep = nms_boxes(boxes, class_scores, iou_thres)
    return boxes[keep], class_ids[keep], class_scores[keep]

def check_violation_and_alert(predictions):
    """检查违规并报警"""
    global no_helmet_flag
    if predictions is None:
        return

    helmet_violation = any(cls_id == 1 for cls_id, _, _ in predictions)
    vest_violation = any(cls_id == 3 for cls_id, _, _ in predictions)

    if helmet_violation or vest_violation:
        current_time = datetime.now()
        last_time = last_alert_time.get("last_time", None)

        if last_time is None or (current_time - last_time).total_seconds() > 5:
            no_helmet_flag = True
            last_alert_time["last_time"] = current_time
            alert_msg = "警告！检测到"
            if helmet_violation and vest_violation:
                alert_msg += "未佩戴头盔和未穿反光衣"
            elif helmet_violation:
                alert_msg += "未佩戴头盔"
            else:
                alert_msg += "未穿反光衣"
            log(f"语音报警: {alert_msg}")
    else:
        no_helmet_flag = False

def draw_boxes(image, predictions):
    """绘制检测框"""
    if predictions is None:
        return image
        
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype("simhei.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for cls_id, score, (x1, y1, x2, y2) in predictions:
        # 确保坐标在图像范围内
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1 = max(0, min(x1, image.shape[1]-1))
        y1 = max(0, min(y1, image.shape[0]-1))
        x2 = max(0, min(x2, image.shape[1]-1))
        y2 = max(0, min(y2, image.shape[0]-1))
        
        colors = {
            0: (0, 255, 0),   # 佩戴头盔 - 绿色
            1: (0, 0, 255),    # 未佩戴头盔 - 红色
            2: (0, 255, 255),  # 穿反光衣 - 黄色
            3: (255, 0, 0)     # 未穿反光衣 - 蓝色
        }
        color = colors.get(cls_id, (255, 255, 255))
        label = f"{class_names[cls_id]} {score:.2f}"
        
        # 绘制矩形框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # 计算文本位置
        try:
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except:
            text_width, text_height = draw.textsize(label, font=font)
        
        # 绘制文本背景
        draw.rectangle(
            [x1, y1 - text_height - 2, 
             x1 + text_width + 2, y1],
            fill=color
        )
        
        # 绘制文本
        draw.text((x1 + 1, y1 - text_height - 1), 
                 label, font=font, fill=(0, 0, 0))
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def voice_monitor():
    """语音播报监控"""
    global voice_process, no_helmet_flag
    while True:
        with speech_lock:
            if no_helmet_flag:
                if voice_process is None or voice_process.poll() is not None:
                    voice_process = subprocess.Popen(
                        ['espeak-ng', '-v', 'zh', "请佩戴安全头盔和反光衣"]
                    )
        time.sleep(0.1)

def detect_image(file_path):
    """图片检测"""
    if model is None:
        messagebox.showwarning("警告", "请先加载RKNN模型！")
        return

    img = cv2.imread(file_path)
    if img is None:
        messagebox.showerror("错误", "图片读取失败！")
        return

    try:
        orig_h, orig_w = img.shape[:2]
        img_resized = cv2.resize(img, (320, 320))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_input = np.expand_dims(img_rgb, axis=0)
        
        output = model.inference(inputs=[img_input])[0]
        boxes, classes, scores = yolov5_post_process(output)
        
        if boxes is not None:
            # 坐标转换回原始尺寸
            scale_x = orig_w / 320
            scale_y = orig_h / 320
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            
            preds = [(classes[i], scores[i], boxes[i]) for i in range(len(boxes))]
            check_violation_and_alert(preds)
            result_img = draw_boxes(img, preds)
        else:
            result_img = img
            log("未检测到目标")

        # 显示结果
        img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        max_display_size = 600
        w, h = img_pil.size
        scale = min(max_display_size / w, max_display_size / h, 1)
        new_size = (int(w * scale), int(h * scale))

        img_pil = img_pil.resize(new_size, Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)

        panel.config(image=img_tk)
        panel.image = img_tk
    except Exception as e:
        log(f"图片检测出错: {e}")
        messagebox.showerror("错误", f"图片检测出错: {e}")

def video_capture_thread(video_path):
    """视频捕获线程"""
    global video_running, video_capture
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        log("视频打开失败！")
        return

    video_running = True
    original_fps = video_capture.get(cv2.CAP_PROP_FPS)
    target_fps = min(original_fps, 30)
    frame_interval = 1.0 / target_fps
    log(f"原始视频帧率: {original_fps}, 目标处理帧率: {target_fps}")

    last_frame_time = time.time()

    while video_running:
        current_time = time.time()
        if current_time - last_frame_time < frame_interval:
            time.sleep(0.001)
            continue

        ret, frame = video_capture.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        try:
            if frame_queue.full():
                frame_queue.get_nowait()
            frame_queue.put(frame)
        except Exception as e:
            log(f"视频队列异常: {e}")

        last_frame_time = current_time

    video_capture.release()
    log("视频读取线程结束")

def inference_worker():
    """推理工作线程"""
    global video_running
    target_fps = 20
    frame_interval = 1.0 / target_fps
    last_infer_time = 0

    while video_running:
        current_time = time.time()
        if current_time - last_infer_time < frame_interval:
            time.sleep(0.001)
            continue

        if frame_queue.empty():
            time.sleep(0.001)
            continue

        frame = frame_queue.get()
        last_infer_time = current_time

        # 预处理
        img_resized = cv2.resize(frame, (320, 320))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_input = np.expand_dims(img_rgb, axis=0)

        try:
            # 推理
            output = model.inference(inputs=[img_input])[0]
            
            # 后处理
            boxes, classes, scores = yolov5_post_process(output)
            
            if boxes is not None:
                # 坐标转换回原始尺寸
                h, w = frame.shape[:2]
                scale_x = w / 320
                scale_y = h / 320
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
                
                # 组合预测结果
                preds = [(classes[i], scores[i], boxes[i]) for i in range(len(boxes))]
                
                # 检查违规
                check_violation_and_alert(preds)
                
                # 绘制结果
                result_img = draw_boxes(frame, preds)
            else:
                result_img = frame
            
            # 放入结果队列
            if not result_queue.full():
                result_queue.put(result_img)
                
        except Exception as e:
            log(f"推理异常: {e}")
            if not result_queue.full():
                result_queue.put(frame)

def update_video_frame():
    """更新视频帧"""
    if not video_running:
        return

    try:
        if not result_queue.empty():
            frame = result_queue.get()
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            max_display_size = 600
            w, h = img_pil.size
            scale = min(max_display_size / w, max_display_size / h, 1)
            new_size = (int(w * scale), int(h * scale))

            img_pil = img_pil.resize(new_size, Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            panel.config(image=img_tk)
            panel.image = img_tk
    except Exception as e:
        log(f"更新视频帧异常: {e}")

    window.after(10, update_video_frame)

def detect_video(video_path):
    """视频检测"""
    global video_thread, inference_thread, video_running

    if model is None:
        messagebox.showwarning("警告", "请先加载RKNN模型！")
        return

    if video_thread and video_thread.is_alive():
        log("已有视频检测在运行，请先停止")
        return

    # 清空队列
    while not frame_queue.empty():
        frame_queue.get()
    while not result_queue.empty():
        result_queue.get()

    video_running = True

    video_thread = threading.Thread(
        target=video_capture_thread, 
        args=(video_path,), 
        daemon=True
    )
    inference_thread = threading.Thread(
        target=inference_worker, 
        daemon=True
    )

    video_thread.start()
    inference_thread.start()

    update_video_frame()
    log(f"开始视频检测: {video_path}")

def detect_camera():
    """摄像头检测"""
    global stop_flag
    if model is None:
        messagebox.showwarning("警告", "请先加载RKNN模型！")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("错误", "无法打开摄像头！")
        return

    stop_flag = False
    log("开始摄像头检测，按q退出")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            if stop_flag:
                log("摄像头检测被停止")
                break

            ret, frame = cap.read()
            if not ret:
                log("摄像头读取失败，退出")
                break

            # 预处理
            img_resized = cv2.resize(frame, (320, 320))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_input = np.expand_dims(img_rgb, axis=0)

            # 推理
            output = model.inference(inputs=[img_input])[0]
            
            # 后处理
            boxes, classes, scores = yolov5_post_process(output)
            
            if boxes is not None:
                # 坐标转换
                h, w = frame.shape[:2]
                scale_x = w / 320
                scale_y = h / 320
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
                
                # 组合预测结果
                preds = [(classes[i], scores[i], boxes[i]) for i in range(len(boxes))]
                
                # 检查违规
                check_violation_and_alert(preds)
                
                # 绘制结果
                result_img = draw_boxes(frame, preds)
            else:
                result_img = frame

            # 显示结果
            cv2.imshow("摄像头检测 - 按q退出", result_img)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                log("摄像头检测被用户中断")
                break
    except Exception as e:
        log(f"摄像头检测出错: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        log("摄像头检测结束")

def load_model():
    """加载RKNN模型"""
    global model
    file_path = filedialog.askopenfilename(filetypes=[("RKNN模型", "*.rknn")])
    if not file_path:
        return

    try:
        model = RKNNLite()
        ret = model.load_rknn(file_path)
        if ret != 0:
            raise Exception(f"加载模型失败: {ret}")
        ret = model.init_runtime()
        if ret != 0:
            raise Exception(f"初始化运行环境失败: {ret}")
        log(f"模型加载成功: {file_path}")
        messagebox.showinfo("成功", "模型加载成功！")
    except Exception as e:
        log(f"模型加载失败: {e}")
        messagebox.showerror("错误", f"模型加载失败: {e}")

def choose_image():
    """选择图片"""
    file_path = filedialog.askopenfilename(filetypes=[("图片文件", "*.jpg *.jpeg *.png")])
    if file_path:
        log(f"选择图片: {file_path}")
        threading.Thread(target=detect_image, args=(file_path,), daemon=True).start()

def choose_video():
    """选择视频"""
    global video_file_path
    file_path = filedialog.askopenfilename(filetypes=[("视频文件", "*.mp4 *.avi *.mov")])
    if file_path:
        log(f"选择视频: {file_path}")
        video_file_path = file_path
        threading.Thread(target=detect_video, args=(file_path,), daemon=True).start()

def start_camera():
    """启动摄像头"""
    threading.Thread(target=detect_camera, daemon=True).start()

def stop_detection():
    """停止检测"""
    global video_running, stop_flag
    stop_flag = True
    video_running = False
    log("停止检测")

def clear_log():
    """清空日志"""
    log_text.delete(1.0, tk.END)
    log("日志已清空")

# 创建主窗口
window = tk.Tk()
window.title("煤矿安全防护装备检测系统")
window.geometry("900x900")
window.configure(bg=BG_COLOR)

# 设置窗口图标
try:
    window.iconbitmap("icon.ico")
except:
    pass

# 标题
title_frame = tk.Frame(window, bg=BG_COLOR)
title_frame.pack(pady=10)
tk.Label(title_frame, text="煤矿安全防护装备检测系统", font=FONT_TITLE, bg=BG_COLOR, fg=TEXT_COLOR).pack()

# 帧率显示
fps_frame = tk.Frame(window, bg=BG_COLOR)
fps_frame.pack()
fps_label = tk.Label(fps_frame, text="帧率: 0.0 FPS", font=FONT_BUTTON, bg=BG_COLOR, fg=TEXT_COLOR)
fps_label.pack()

# 控制按钮区域
control_frame = tk.Frame(window, bg=BG_COLOR)
control_frame.pack(pady=10)

# 按钮样式配置
style = ttk.Style()
style.configure('TButton', 
                font=FONT_BUTTON,
                borderwidth=1,
                relief='raised',
                background=BUTTON_COLOR,
                foreground=TEXT_COLOR)
style.map('TButton',
          background=[('active', '#2980b9')],
          relief=[('pressed', 'sunken')])

# 第一行按钮
btn1 = ttk.Button(control_frame, text="加载RKNN模型", command=load_model, width=20)
btn1.grid(row=0, column=0, padx=5, pady=5)
btn2 = ttk.Button(control_frame, text="图片识别", command=choose_image, width=20)
btn2.grid(row=0, column=1, padx=5, pady=5)
btn3 = ttk.Button(control_frame, text="视频识别", command=choose_video, width=20)
btn3.grid(row=0, column=2, padx=5, pady=5)

# 第二行按钮
btn4 = ttk.Button(control_frame, text="摄像头识别", command=start_camera, width=20)
btn4.grid(row=1, column=0, padx=5, pady=5)
btn5 = ttk.Button(control_frame, text="停止检测", command=stop_detection, width=20)
btn5.grid(row=1, column=1, padx=5, pady=5)
btn6 = ttk.Button(control_frame, text="清空日志", command=clear_log, width=20)
btn6.grid(row=1, column=2, padx=5, pady=5)

# 图片显示区域
panel_frame = tk.Frame(window, bg=PANEL_BG, bd=2, relief=tk.SUNKEN)
panel_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
panel = tk.Label(panel_frame, bg=PANEL_BG)
panel.pack(pady=10, padx=10)

# 日志区域
log_frame = tk.Frame(window, bg=BG_COLOR)
log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

tk.Label(log_frame, text="系统日志:", font=FONT_BUTTON, bg=BG_COLOR, fg=TEXT_COLOR).pack(anchor=tk.W)

log_text = tk.Text(log_frame, height=10, wrap=tk.WORD, bg=LOG_BG, fg=LOG_FG, font=FONT_LOG)
log_text.pack(fill=tk.BOTH, expand=True)

scrollbar = ttk.Scrollbar(log_text)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
log_text.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=log_text.yview)

# 启动帧率计算和语音监控
window.after(100, update_fps)
threading.Thread(target=voice_monitor, daemon=True).start()

window.mainloop()
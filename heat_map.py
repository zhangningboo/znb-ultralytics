import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw # 用于图像处理和在占位符上绘制文本
import matplotlib.colors
import cv2 # OpenCV 用于高斯模糊

# 尝试导入YOLO模型库
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: 'ultralytics' library not found. YOLO model functionality will be unavailable, falling back to simulated heatmap.")
    print("Please install the required libraries with 'pip install ultralytics opencv-python'.")
    YOLO_AVAILABLE = False

# --- 辅助函数：生成模拟热力图 (用作占位符或回退) ---
def generate_simulated_heatmap(image_width, image_height, object_center_x, object_center_y, object_width, object_height, max_intensity=255, falloff_rate=0.0005):
    """
    Generates a simulated heatmap for an object.
    Intensity is highest at the object's center and falls off.
    This function serves as a placeholder for actual model output or as a fallback.
    """
    y, x = np.ogrid[:image_height, :image_width]
    std_x = object_width / 2
    std_y = object_height / 2
    std_x = max(std_x, 1) # Avoid division by zero
    std_y = max(std_y, 1) # Avoid division by zero

    dist_sq = (((x - object_center_x)**2) / (2 * std_x**2)) + \
              (((y - object_center_y)**2) / (2 * std_y**2))
    heatmap = max_intensity * np.exp(-dist_sq * falloff_rate * 10)
    return np.clip(heatmap, 0, max_intensity)

# --- 函数：从真实模型获取热力图 ---
def get_heatmap_from_actual_model(image_np, model_type='detection', object_class_name='cup'):
    """
    Attempts to get a heatmap from a real model.
    Uses YOLOv10x if available for object detection and heatmap generation.
    Otherwise, falls back to a simulated heatmap.

    Args:
        image_np (numpy.ndarray): Input image as a NumPy array (H, W, C).
        model_type (str): Currently only 'detection' is supported.
        object_class_name (str): Target class name for detection (e.g., 'cup').

    Returns:
        numpy.ndarray: Generated heatmap (2D array).
    """
    print(f"Attempting to generate heatmap using '{model_type}' model approach.")
    image_height, image_width = image_np.shape[:2]

    if model_type == 'detection' and YOLO_AVAILABLE:
        try:
            model_name = 'yolo11n.pt'
            print(f"  Step: Loading pre-trained {model_name} model.")
            model = YOLO(model_name)
            print("  Step: Preprocessing image and performing inference.")
            # 可以调整推理参数，例如置信度阈值 conf
            results = model(image_np, verbose=False, conf=0.25) # verbose=False, 增加conf参数示例

            heatmap = np.zeros((image_height, image_width), dtype=np.float32)
            detections_found = 0

            print(f"  Step: Filtering for '{object_class_name}' class detections.")
            target_cls_id = -1
            if hasattr(model, 'names') and isinstance(model.names, dict):
                for cls_id, name_val in model.names.items(): # Renamed 'name' to 'name_val' to avoid conflict
                    if name_val == object_class_name:
                        target_cls_id = cls_id
                        break
            else:
                print(f"  Warning: Model class names (model.names) not available in the expected format. Cannot map '{object_class_name}' to class ID.")


            if target_cls_id == -1:
                print(f"  Warning: Class '{object_class_name}' not found in model's classes or model.names not accessible. Will display an empty heatmap.")
            else:
                print(f"  Class ID for '{object_class_name}': {target_cls_id}")

                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls)
                        conf = float(box.conf)
                        if cls == target_cls_id:
                            detections_found += 1
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            # 使用置信度作为热度值填充矩形
                            cv2.rectangle(heatmap, (x1, y1), (x2, y2), conf, thickness=cv2.FILLED)

                if detections_found > 0:
                    print(f"  Found {detections_found} '{object_class_name}' detection(s).")
                    # 调整高斯模糊的核大小，可以根据效果调整
                    # 较大的核会产生更模糊（弥散）的热力图
                    blur_kernel_size = (101, 101) # 可以尝试减小如 (51,51) 或增大
                    heatmap = cv2.GaussianBlur(heatmap, blur_kernel_size, 0)
                    if heatmap.max() > 0:
                        heatmap = (heatmap / heatmap.max()) * 255 # 归一化到0-255
                    print("  Step: Heatmap generated based on detections.")
                    return heatmap.astype(np.uint8)
                else:
                    print(f"  No '{object_class_name}' detections found with current settings. Will display an empty heatmap.")
                    return heatmap # Return empty heatmap

        except Exception as e:
            print(f"  Error during YOLO model operation: {e}")
            print("  Fallback: Using simulated heatmap.")
            # Fallthrough to simulated heatmap generation

    # ----- Fallback to simulated heatmap if model is unavailable or an error occurs -----
    print("  Fallback: Using simulated heatmap as a placeholder.")
    center_x_ratio = 0.47
    center_y_ratio = 0.45
    width_ratio = 0.20
    height_ratio = 0.30

    obj_center_x_abs = int(center_x_ratio * image_width)
    obj_center_y_abs = int(center_y_ratio * image_height)
    obj_width_abs = int(width_ratio * image_width)
    obj_height_abs = int(height_ratio * image_height)

    simulated_heatmap = generate_simulated_heatmap(
        image_width, image_height,
        obj_center_x_abs, obj_center_y_abs,
        obj_width_abs, obj_height_abs
    )
    return simulated_heatmap

def plot_image_with_heatmap(image_path, heatmap_data, title="Object Detection Heatmap", alpha=0.6, cmap_name='inferno'):
    """
    Overlays a heatmap on an image and displays it. All plot text is in English.
    """
    try:
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}.")
        img = Image.new('RGB', (500, 500), color = (128, 128, 128))
        draw = ImageDraw.Draw(img)
        draw.text((50, 230), "Image not found.\nPlease use a valid path.", fill=(255,0,0))
        heatmap_data = np.zeros((500, 500))
        print("Displaying placeholder image and empty heatmap.")

    img_np = np.array(img)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(img_np)

    if heatmap_data.max() > 0:
        if heatmap_data.shape[0] != img_np.shape[0] or heatmap_data.shape[1] != img_np.shape[1]:
            print(f"Warning: Heatmap dimensions ({heatmap_data.shape}) differ from image dimensions ({img_np.shape[:2]}). Resizing heatmap.")
            heatmap_pil = Image.fromarray(heatmap_data.astype(np.uint8))
            heatmap_resized_pil = heatmap_pil.resize((img_np.shape[1], img_np.shape[0]), Image.BICUBIC)
            heatmap_data_resized = np.array(heatmap_resized_pil)
            cax = ax.imshow(heatmap_data_resized, cmap=plt.get_cmap(cmap_name), alpha=alpha, extent=(0, img_np.shape[1], img_np.shape[0], 0))
        else:
            cax = ax.imshow(heatmap_data, cmap=plt.get_cmap(cmap_name), alpha=alpha, extent=(0, img_np.shape[1], img_np.shape[0], 0))

        cbar = fig.colorbar(cax, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Heatmap Intensity (Model-derived or Simulated)', rotation=270, labelpad=15)
    else:
        print("Heatmap is empty (no detections or model not run), not overlaying.")

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("X-coordinate (pixels)", fontsize=12)
    ax.set_ylabel("Y-coordinate (pixels)", fontsize=12)
    ax.axis('on')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # --- Configuration ---
    image_file_path = 'MVI_39781_01408.jpg' # 默认使用您提到识别有困难的俯视图图像
    # image_file_path = 'image_2d8ceb.png' # 之前可以识别的图像
    # image_file_path = 'image_2d208d.jpg' # 另一张测试图像

    target_object_name = 'car'

    # --- 加载图像 ---
    try:
        img_for_model = Image.open(image_file_path).convert('RGB')
        img_np_for_model = np.array(img_for_model)
        img_height, img_width = img_np_for_model.shape[:2]
        print(f"Preparing to generate heatmap for image: {image_file_path} (Dimensions: {img_width}x{img_height})")
    except FileNotFoundError:
        print(f"Fatal Error: Image file '{image_file_path}' not found. Cannot proceed.")
        img_np_for_model = np.zeros((500, 500, 3), dtype=np.uint8)
        img_width, img_height = 500, 500


    # --- Generate Heatmap ---
    heatmap_output = get_heatmap_from_actual_model(
        img_np_for_model,
        model_type='detection',
        object_class_name=target_object_name
    )

    # --- Plot Image with Heatmap ---
    plot_title = f"Heatmap for '{target_object_name}' (YOLOv10x or Simulated)"
    plot_image_with_heatmap(
        image_path=image_file_path,
        heatmap_data=heatmap_output,
        title=plot_title,
        alpha=0.5,
        cmap_name='inferno'
    )

    if not YOLO_AVAILABLE:
        print("\nReminder: To use the actual YOLO model for heatmap generation, ensure 'ultralytics' and 'opencv-python' are installed.")
        print("You can install them via 'pip install ultralytics opencv-python'.")
        print("Currently displaying a simulated heatmap.")
import numpy as np
import os
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import tensorflow as tf
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei
import struct
# ------------------------- 模型加载 -------------------------
def load_model(model_path):
    """加载保存的Keras模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
    return tf.keras.models.load_model(model_path)

# ------------------------- GUI界面 -------------------------
class DigitRecognizer:
    def __init__(self, model):
        self.model = model
        self.window = tk.Tk()
        self.window.title("手写数字识别系统 v2.0")

        # 界面布局
        self.setup_ui()

        # 初始化绘图变量
        self.image = Image.new("L", (280, 280), 0)  # 创建灰度图像
        self.draw = ImageDraw.Draw(self.image)
        self.last_point = None

    def setup_ui(self):
        """构建用户界面"""
        # 左侧绘图区域
        self.canvas = tk.Canvas(self.window, width=280, height=280, bg="white")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        # 右侧控制面板
        control_frame = tk.Frame(self.window)
        control_frame.grid(row=0, column=1, sticky="n")

        # 功能按钮
        tk.Button(control_frame, text="识别", command=self.predict, width=15, height=2).pack(pady=5)
        tk.Button(control_frame, text="清空画板", command=self.clear_canvas, width=15, height=2).pack(pady=5)
        tk.Button(control_frame, text="退出系统", command=self.window.quit, width=15, height=2).pack(pady=5)

        # 结果显示区域
        self.result_label = tk.Label(control_frame, text="等待识别...", font=("SimHei", 16), fg="blue")
        self.result_label.pack(pady=20)

        # 绑定绘图事件
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last_point)

    def paint(self, event):
        """处理鼠标绘图事件"""
        if self.last_point:
            x1, y1 = self.last_point
            x2, y2 = event.x, event.y
            self.canvas.create_line(x1, y1, x2, y2, width=15, capstyle=tk.ROUND, smooth=True)
            self.draw.line([x1, y1, x2, y2], fill=255, width=15)
        self.last_point = (event.x, event.y)

    def reset_last_point(self, event):
        """重置连线点"""
        self.last_point = None

    def clear_canvas(self):
        """清空画板"""
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="画板已清空", fg="gray")

    def preprocess_image(self):
        # plt.imshow(self.image, cmap='gray')
        # plt.title("反转 Image")
        # plt.show()

        """图像预处理"""
        # 1. 缩放到28x28
        small_img = self.image.resize((28, 28))

        # plt.imshow(small_img, cmap='gray')
        # plt.title("缩放 Image")
        # plt.show()

        # 2. 转换为numpy数组并归一化
        img_array = np.array(small_img) / 255.0

        # plt.imshow(img_array, cmap='gray')
        # plt.title("归一 Image")
        # plt.show()

        # 3. 反转为白底黑字(无需反转)
        # img_array = 1 - img_array
        # plt.imshow(img_array, cmap='gray')
        # plt.title("反转 Image")
        # plt.show()

        # 4. 展平为(1, 784)
        processed_img = img_array.flatten().reshape(1, 784)
        return processed_img

    def predict(self):
        """执行预测"""
        try:
            # 预处理图像
            processed_img = self.preprocess_image()

            # 模型预测
            probabilities = self.model.predict(processed_img)
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[0][predicted_class]

            # 显示结果
            self.show_prediction(predicted_class, confidence)
        except Exception as e:
            self.result_label.config(text=f"错误: {str(e)}", fg="red")

    def show_prediction(self, number, confidence):
        """可视化显示预测结果"""
        color = "green" if confidence > 0.8 else "orange"
        self.result_label.config(text=f"预测数字: {number}\n置信度: {confidence:.2%}", fg=color)

    def run(self):
        """启动主循环"""
        self.window.mainloop()

# ------------------------- 主程序 -------------------------
if __name__ == "__main__":
    MODEL_PATH = "modules/256+256+128+128+64_30_64_relu_0.98250_0318_1656.keras"  # 模型路径

    try:
        print("正在加载模型...")
        model = load_model(MODEL_PATH)
        app = DigitRecognizer(model)
        print("系统启动成功！")
        app.run()
    except Exception as e:
        print(f"系统初始化失败: {str(e)}")
        print("可能原因：")
        print("1. 模型路径错误或文件缺失")
        print("2. 模型文件与当前代码版本不兼容")
        print("3. 缺少依赖库")

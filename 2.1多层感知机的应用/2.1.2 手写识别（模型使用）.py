import numpy as np
import os
import tkinter as tk
import datetime
from PIL import Image, ImageDraw, ImageTk

# ------------------------- 激活函数 -------------------------
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(a):
    return 1 - a**2

# ------------------------- 模型加载 -------------------------
class DynamicMLP:
    def __init__(self, layer_sizes, activation):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.weights = []
        self.biases = []

        # 初始化参数（仅定义结构，实际参数由加载函数填充）
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.zeros(1))  # 占位符
            self.biases.append(np.zeros(1))  # 占位符

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.weights)-1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            if self.activation == 'relu':
                a = relu(z)
            elif self.activation == 'sigmoid':
                a = sigmoid(z)
            elif self.activation == 'tanh':
                a = tanh(z)
            self.activations.append(a)

        # 输出层（Softmax）
        z_output = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        exp_z = np.exp(z_output - np.max(z_output, axis=1, keepdims=True))
        self.activations.append(exp_z / np.sum(exp_z, axis=1, keepdims=True))
        return self.activations[-1]

def load_model(model_dir):
    config_path = os.path.join(model_dir, "config.npy")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到模型配置文件: {config_path}")

    config = np.load(config_path, allow_pickle=True).item()
    model = DynamicMLP(config['layers'], config['activation'])

    # 加载各层参数
    for i in range(len(config['layers']) - 1):
        w_path = os.path.join(model_dir, f"w_{i}.npy")
        b_path = os.path.join(model_dir, f"b_{i}.npy")
        if not os.path.exists(w_path) or not os.path.exists(b_path):
            raise FileNotFoundError(f"缺失模型参数文件: w_{i}.npy 或 b_{i}.npy")

        model.weights[i] = np.load(w_path)
        model.biases[i] = np.load(b_path)

    print(f"模型结构: {config['layers']}")
    print(f"权重形状: {[w.shape for w in model.weights]}")
    print(f"偏置形状: {[b.shape for b in model.biases]}")
    return model


# ------------------------- GUI界面 -------------------------
class DigitRecognizer:
    def __init__(self, model):
        self.model = model
        self.window = tk.Tk()
        self.window.title("手写数字识别系统 v1.0")

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
        tk.Button(control_frame, text="识别", command=self.predict,
                  width=15, height=2).pack(pady=5)
        tk.Button(control_frame, text="清空画板", command=self.clear_canvas,
                  width=15, height=2).pack(pady=5)
        tk.Button(control_frame, text="退出系统", command=self.window.quit,
                  width=15, height=2).pack(pady=5)

        # 结果显示区域
        self.result_label = tk.Label(control_frame, text="等待识别...",
                                     font=("微软雅黑", 16), fg="blue")
        self.result_label.pack(pady=20)

        # 绑定绘图事件
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last_point)

    def paint(self, event):
        """处理鼠标绘图事件"""
        # 平滑连线处理
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
        """图像预处理"""
        # 1. 缩放到28x28
        small_img = self.image.resize((28, 28))
        # # 检查出错问题用 保存预处理后的图像（无误）
        # small_img.save("debug_input.png")

        # 2. 转换为numpy数组并归一化
        img_array = np.array(small_img) / 255.0

        # 3. 颜色反转（与MNIST数据格式一致）
        img_array = 1 - img_array

        # 4. 展平为(1, 784)
        return img_array.flatten().reshape(1, 784)

    def predict(self):
        """执行预测"""
        try:
            # 预处理图像
            processed_img = self.preprocess_image()

            # 模型预测
            probabilities = self.model.forward(processed_img)
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[0][predicted_class]

            # 显示结果
            self.show_prediction(predicted_class, confidence)
        except Exception as e:
            self.result_label.config(text=f"错误: {str(e)}", fg="red")

    def show_prediction(self, number, confidence):
        """可视化显示预测结果"""
        color = "green" if confidence > 0.8 else "orange"
        self.result_label.config(
            text=f"预测数字: {number}\n置信度: {confidence:.2%}",
            fg=color
        )

    def run(self):
        """启动主循环"""
        self.window.mainloop()

# ------------------------- 主程序 -------------------------
if __name__ == "__main__":

    # 模型路径配置
    MODEL_DIR = "modules\\model_20250308_160120"  # 模型路径

    try:
        # 加载模型
        print("正在加载模型...")
        model = load_model(MODEL_DIR)

        # 启动GUI
        app = DigitRecognizer(model)
        print("系统启动成功！")
        app.run()
    except Exception as e:
        print(f"系统初始化失败: {str(e)}")
        print("可能原因：")
        print("1. 模型路径错误或文件缺失")
        print("2. 模型文件与当前代码版本不兼容")
        print("3. 缺少依赖库")
import numpy as np
import os
import tkinter as tk
import datetime
from PIL import Image, ImageDraw, ImageTk
import matplotlib

matplotlib.use('TkAgg')  # 强制使用 TkAgg 后端（pycharm问题）
import matplotlib.pyplot as plt
from io import BytesIO
import struct


# ------------------------- 数据加载（调试用，用于确定模型对正确数据输入识别正确） -------------------------
def load_mnist(data_dir="data/mnist"):
    import gzip
    import shutil

    # 确保目录存在
    os.makedirs(data_dir, exist_ok=True)

    # 检查是否已解压
    required_files = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte"
    ]

    # 解压.gz文件（如果未解压）
    for file in required_files:
        gz_path = os.path.join(data_dir, file + ".gz")
        bin_path = os.path.join(data_dir, file)

        if not os.path.exists(bin_path):
            if os.path.exists(gz_path):
                print(f"解压文件中: {file}.gz")
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(bin_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                raise FileNotFoundError(f"找不到文件: {gz_path}")

    # 加载数据
    def read_images(path):
        with open(path, 'rb') as f:
            _, num, rows, cols = struct.unpack(">IIII", f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols) / 255.0

    def read_labels(path):
        with open(path, 'rb') as f:
            _, num = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    return (
        read_images(os.path.join(data_dir, "train-images-idx3-ubyte")),
        read_labels(os.path.join(data_dir, "train-labels-idx1-ubyte")),
        read_images(os.path.join(data_dir, "t10k-images-idx3-ubyte")),
        read_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte"))
    )


def test_modules_real():
    # 加载MNIST数据集
    X_train, y_train, X_test, y_test = load_mnist()

    # ------------------------- 测试与可视化 -------------------------
    # 随机选取几个测试样本
    num_samples = 15
    sample_indices = np.random.choice(len(X_test), num_samples, replace=False)

    # 预测并可视化
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))  # 3行5列
    for i, idx in enumerate(sample_indices):
        # 获取样本
        img = X_test[idx].reshape(28, 28)
        true_label = y_test[idx]

        # 预测
        prob = model.forward(img.flatten().reshape(1, -1))
        predicted_label = np.argmax(prob)
        confidence = prob[0][predicted_label]

        # 可视化
        ax = axes[i // 5, i % 5]  # 计算子图位置
        ax.imshow(img, cmap='gray')
        ax.set_title(f"True: {true_label}\nPred: {predicted_label} ({confidence:.2%})")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


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
    return 1 - a ** 2


# ------------------------- 模型加载 -------------------------
class DynamicMLP:
    def __init__(self, layer_sizes, activation):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.weights = []
        self.biases = []

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.weights) - 1):
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

        model.weights.append(np.load(w_path))
        model.biases.append(np.load(b_path))

    # 调试：打印加载的权重和偏置形状
    print(f"模型结构: {config['layers']}")
    print(f"权重形状: {[w.shape for w in model.weights]}")
    print(f"偏置形状: {[b.shape for b in model.biases]}")
    return model


# ------------------------- GUI界面 -------------------------
class DigitRecognizer:
    def __init__(self, model):
        self.model = model
        self.window = tk.Tk()
        self.window.title("手写数字识别系统 v1.1")  # (禁用反转后输入数据正确可以预测)

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
                                     font=("SimHei", 16), fg="blue")
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

        # # 调试：保存缩放后的图像
        # small_img.save("debug_scaled.png")

        # 2. 转换为numpy数组并归一化
        img_array = np.array(small_img) / 255.0

        # # 调试：保存归一化后的图像
        # import matplotlib.pyplot as plt
        # plt.imshow(img_array, cmap='gray')
        # plt.title("Normalized Image")
        # plt.savefig("debug_normalized.png")
        # plt.show()

        # # 3. 颜色反转（与MNIST数据格式一致）（经过测试发现不需要反转）
        # img_array = 1 - img_array

        # # 调试：保存颜色反转后的图像
        # plt.imshow(img_array, cmap='gray')
        # plt.title("Inverted Image")
        # plt.savefig("debug_inverted.png")
        # plt.show()

        # 4. 展平为(1, 784)
        processed_img = img_array.flatten().reshape(1, 784)

        # # 调试：打印预处理后的图像数据
        # print(f"预处理后的图像数据形状: {processed_img.shape}")
        # print(f"预处理后的图像数据: {processed_img[:10]}")  # 打印前10个值

        return processed_img

    def predict(self):
        """执行预测"""
        try:
            # 预处理图像
            processed_img = self.preprocess_image()

            # 调试：打印预处理后的图像数据
            print(f"预处理后的图像数据形状: {processed_img.shape}")
            print(f"预处理后的图像数据: {processed_img[:10]}")  # 打印前10个值

            # 模型预测
            probabilities = self.model.forward(processed_img)

            # 调试：打印模型输出
            print(f"模型输出: {probabilities}")

            predicted_class = np.argmax(probabilities)
            confidence = probabilities[0][predicted_class]

            # 调试：打印预测结果
            print(f"预测数字: {predicted_class}, 置信度: {confidence:.2%}")

            # 可视化输入图像
            import matplotlib.pyplot as plt
            img_array = processed_img.reshape(28, 28)
            plt.imshow(img_array, cmap='gray')
            plt.title(f"输入图像\n预测数字: {predicted_class}, 置信度: {confidence:.2%}")
            plt.show()

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
    MODEL_DIR = "modules\\model_20250316_201334"  # 模型路径

    try:
        # 加载模型
        print("正在加载模型...")
        model = load_model(MODEL_DIR)
        # 测试模型是否正确加载，即是否可以正确识别mnist本身的正确数据
        test_modules_real()
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

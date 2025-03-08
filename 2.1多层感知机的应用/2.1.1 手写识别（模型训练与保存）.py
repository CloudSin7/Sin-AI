import datetime
import numpy as np
import os
import struct

# ------------------------- 数据加载 -------------------------
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

    # 加载数据（以下代码保持原样）
    def read_images(path):
        with open(path, 'rb') as f:
            _, num, rows, cols = struct.unpack(">IIII", f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows*cols) / 255.0

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

# ------------------------- 用户输入 -------------------------
def get_user_input():
    print("\n" + "="*40)
    print("=== MNIST MLP 参数设置 ===")
    hidden_layers = input("隐藏层结构（逗号分隔，如128,64）: ") or "128"
    layers = [784] + list(map(int, hidden_layers.split(','))) + [10]

    params = {
        'layers': layers,
        'epochs': int(input("训练轮数 (默认20): ") or 20),
        'batch_size': int(input("批量大小 (默认64): ") or 64),
        'learning_rate': float(input("学习率 (默认0.1): ") or 0.1),
        'activation': input("激活函数（relu/sigmoid/tanh，默认relu）: ").lower() or 'relu'
    }
    return params

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

# ------------------------- 动态MLP模型 -------------------------
class DynamicMLP:
    def __init__(self, layer_sizes, activation):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.weights = []
        self.biases = []

        # 初始化参数
        for i in range(len(layer_sizes)-1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i+1]
            # 根据激活函数选择初始化方法
            if activation == 'relu':
                std = np.sqrt(2.0 / in_size)  # He初始化
            else:
                std = np.sqrt(1.0 / in_size)  # Xavier初始化
            self.weights.append(np.random.randn(in_size, out_size) * std)
            self.biases.append(np.zeros((1, out_size)))

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

    def backward(self, X, y, lr):
        m = X.shape[0]
        gradients_w = []
        gradients_b = []

        # 输出层梯度
        delta = self.activations[-1].copy()
        delta[np.arange(m), y] -= 1
        delta /= m

        for i in reversed(range(len(self.weights))):
            a_prev = self.activations[i]
            grad_w = np.dot(a_prev.T, delta)
            grad_b = np.sum(delta, axis=0, keepdims=True)
            gradients_w.append(grad_w)
            gradients_b.append(grad_b)

            if i > 0:
                if self.activation == 'relu':
                    delta = np.dot(delta, self.weights[i].T) * relu_derivative(self.z_values[i-1])
                elif self.activation == 'sigmoid':
                    delta = np.dot(delta, self.weights[i].T) * sigmoid_derivative(self.activations[i])
                elif self.activation == 'tanh':
                    delta = np.dot(delta, self.weights[i].T) * tanh_derivative(self.activations[i])

        # 更新参数
        for i in range(len(self.weights)):
            self.weights[i] -= lr * gradients_w[::-1][i]
            self.biases[i] -= lr * gradients_b[::-1][i]

# ------------------------- 训练函数封装 -------------------------
def train_model(model, X_train, y_train, X_test, y_test, params):
    history = {'loss': [], 'accuracy': []}

    for epoch in range(params['epochs']):
        # 数据打乱
        indices = np.random.permutation(len(X_train))
        X_shuffled, y_shuffled = X_train[indices], y_train[indices]

        total_loss = 0
        # 分批训练
        for i in range(0, len(X_train), params['batch_size']):
            # 获取批次数据
            X_batch = X_shuffled[i:i+params['batch_size']]
            y_batch = y_shuffled[i:i+params['batch_size']]

            # 前向传播
            probs = model.forward(X_batch)
            # 计算损失
            batch_loss = -np.mean(np.log(probs[np.arange(len(y_batch)), y_batch]))
            total_loss += batch_loss * len(y_batch)

            # 反向传播
            current_lr = params['learning_rate'] * (0.95 ** epoch)  # 学习率递减，防过度
            model.backward(X_batch, y_batch, current_lr)

        # 计算平均损失
        avg_loss = total_loss / len(X_train)
        history['loss'].append(avg_loss)

        # 测试集评估
        test_probs = model.forward(X_test)
        test_pred = np.argmax(test_probs, axis=1)
        accuracy = np.mean(test_pred == y_test)
        history['accuracy'].append(accuracy)

        # 打印进度
        print(f"Epoch {epoch+1}/{params['epochs']} | Loss: {avg_loss:.4f} | 准确率: {accuracy:.4f}")

    return model, history

# ------------------------- 参数展示 -------------------------
def show_model_params(model):
    print(f"\n{'='*50}\n=== 模型结构 ({'→'.join(map(str, model.layer_sizes))}) ===")
    for i, (w, b) in enumerate(zip(model.weights, model.biases)):
        print(f"层 {i+1} ({w.shape[0]}→{w.shape[1]})")
        print(f"  权重范围: [{w.min():.4f}, {w.max():.4f}]")
        print(f"  偏置范围: [{b.min():.4f}, {b.max():.4f}]")
    print("="*50)

# ------------------------- 保存模型 -------------------------
def save_model(model, save_dir="modules"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(save_dir, f"model_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    # 保存各层参数
    for i, (w, b) in enumerate(zip(model.weights, model.biases)):
        np.save(os.path.join(model_dir, f"w_{i}.npy"), w)
        np.save(os.path.join(model_dir, f"b_{i}.npy"), b)

    # 保存模型配置
    config = {
        'layers': model.layer_sizes,
        'activation': model.activation
    }
    np.save(os.path.join(model_dir, "config.npy"), config)
    print(f"模型已保存至: {model_dir}")

# ------------------------- 主程序 -------------------------
if __name__ == "__main__":
    # 加载数据
    X_train, y_train, X_test, y_test = load_mnist()

    # 用户参数
    params = get_user_input()

    # 初始化模型
    model = DynamicMLP(params['layers'], params['activation'])
    show_model_params(model)

    # 执行训练
    trained_model, history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        params=params
    )

    # 最终参数
    show_model_params(model)

    # 保存模型到本地
    is_save = input("保存到本地(yes/no):")
    if is_save == "yes":
        save_model(model)
    else:
        print("程序结束")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 强制使用 TkAgg 后端
# 数据5000 模型124,124,124  学习率0.01  训练5000 实测可以基本拟合


# 生成训练数据
def generate_data():
    while True:
        try:
            num_samples = int(input("请输入要生成的训练数据量（推荐1000-10000）："))
            if num_samples <= 0:
                raise ValueError
            break
        except ValueError:
            print("请输入一个正整数！")

    x = np.linspace(-2 * np.pi, 2 * np.pi, num_samples)
    y = np.sin(x) + np.random.randn(num_samples) * 0.1  # 添加少量噪声
    return x.reshape(-1, 1), y.reshape(-1, 1)
# 这里用于生成大量随机的且带一定偏移量的正弦数据

# 数据展示
def display_data(x, y, n=5):
    print("\n数据集统计:")
    print(f"样本总数: {len(x)}")
    print(f"输入范围: [{x.min():.2f}, {x.max():.2f}]")
    print(f"输出均值: {y.mean():.2f}，标准差: {y.std():.2f}")
    print("\n前5个样本示例:")
    for i in range(min(n, len(x))):
        print(f"X[{i}]: {x[i][0]:.3f} → Y[{i}]: {y[i][0]:.3f}")
# 大致显示一下数据生成的结果

# Xavier初始化
def xavier_init(size_in, size_out):
    return np.random.randn(size_in, size_out) * np.sqrt(2.0 / (size_in + size_out))
# 之前我们使用的是普通的随机初始化方法，也就是权重大小完全随机，但是这样有一定的问题
# 比如，当我们输入层的参数很少的时候，但是隐藏层神经元很多的时候。或者简单来说前一层参数很少，下一层参数很多的时候
# 使用之前的随机初始化方法就会发现，n+1层的每一个参数都是由n层的所有参数乘以权重相加而后经过激活函数计算得来

class SimpleMLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        # 初始化隐藏层
        self.hidden_layers = []
        prev_size = input_size
        for i, size in enumerate(hidden_sizes):
            self.hidden_layers.append({
                'W': xavier_init(prev_size, size),
                'b': np.zeros((1, size)),
                'activation': self.relu,
                'activation_derivative': self.relu_derivative
            })
            prev_size = size

        # 初始化输出层
        self.output_layer = {
            'W': xavier_init(prev_size, output_size),
            'b': np.zeros((1, output_size))
        }

    # ReLU激活函数及其导数
    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    # 前向传播（保留中间结果）
    def forward(self, X):
        self.cache = [X]  # 存储各层激活值
        current_output = X

        # 计算隐藏层
        for layer in self.hidden_layers:
            z = np.dot(current_output, layer['W']) + layer['b']
            a = layer['activation'](z)
            self.cache.append(a)
            current_output = a

        # 计算输出层
        z_output = np.dot(current_output, self.output_layer['W']) + self.output_layer['b']
        self.cache.append(z_output)
        return z_output

    # 计算损失（含L2正则化）
    def compute_loss(self, predictions, targets):
        mse = np.mean((predictions - targets) ** 2)
        l2_reg = 0.0
        # 计算所有隐藏层的正则项
        for layer in self.hidden_layers:
            l2_reg += np.sum(layer['W'] ** 2)
        # 加上输出层的正则项
        l2_reg += np.sum(self.output_layer['W'] ** 2)
        return mse + 0.01 * l2_reg / (2 * targets.shape[0])

    # 反向传播（修正梯度计算）
    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        gradients = []

        # 输出层梯度
        dZ_output = (self.cache[-1] - y) / m
        dW_output = np.dot(self.cache[-2].T, dZ_output) + 0.01 * self.output_layer['W'] / m
        db_output = np.sum(dZ_output, axis=0, keepdims=True)
        gradients.append((dW_output, db_output))

        # 反向传播隐藏层
        dZ = dZ_output
        for i in reversed(range(len(self.hidden_layers))):
            layer = self.hidden_layers[i]
            a_prev = self.cache[i]

            # 计算当前层的梯度
            dA = np.dot(dZ, self.output_layer['W'].T) if i == len(self.hidden_layers) - 1 \
                else np.dot(dZ, self.hidden_layers[i + 1]['W'].T)
            dZ = dA * layer['activation_derivative'](self.cache[i + 1])

            dW = np.dot(a_prev.T, dZ) + 0.01 * layer['W'] / m
            db = np.sum(dZ, axis=0, keepdims=True)
            gradients.append((dW, db))

        # 更新参数（倒序应用梯度）
        self.output_layer['W'] -= learning_rate * gradients[0][0]
        self.output_layer['b'] -= learning_rate * gradients[0][1]

        for i, (dW, db) in enumerate(reversed(gradients[1:])):
            self.hidden_layers[i]['W'] -= learning_rate * dW
            self.hidden_layers[i]['b'] -= learning_rate * db

    # 训练过程（带进度显示）
    def train(self, X, y, learning_rate=0.01, epochs=1000):
        print("\n训练开始...")
        print_interval = max(1, epochs // 10)  # 自动调整显示间隔

        for epoch in range(epochs):
            predictions = self.forward(X)
            loss = self.compute_loss(predictions, y)
            self.backward(X, y, learning_rate)

            if epoch % print_interval == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:5d}/{epochs} | Loss: {loss:.6f}")


# 交互式预测
def prediction_menu(model):
    print("\n进入预测模式（输入'exit'退出）")
    while True:
        try:
            user_input = input("请输入要预测的X值（建议范围：-6.28~6.28）: ").strip()
            if user_input.lower() == 'exit':
                break

            x_test = float(user_input)
            if not (-6.29 <= x_test <= 6.29):
                print("注意：输入值超出训练数据范围！")

            X_test = np.array([[x_test]])
            prediction = model.forward(X_test)
            print(f"预测结果：sin({x_test:.3f}) ≈ {prediction[0][0]:.4f}")
        except ValueError:
            print("输入无效，请输入数字或'exit'")


# 主程序
if __name__ == "__main__":
    # 生成训练数据
    X_train, y_train = generate_data()
    display_data(X_train, y_train)

    # 模型配置
    print("\n模型配置：")
    hidden_layers = list(map(int, input("请输入隐藏层结构（用逗号分隔，如64,64,64）: ").split(',')))
    epochs = int(input("请输入训练轮次（推荐1000-10000）: "))
    lr = float(input("请输入学习率（推荐0.001-0.1）: "))

    # 创建并训练模型
    mlp = SimpleMLP(input_size=1, hidden_sizes=hidden_layers, output_size=1)
    mlp.train(X_train, y_train, learning_rate=lr, epochs=epochs)

    # 可视化结果
    plt.figure(figsize=(10, 6))
    X_plot = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)
    y_plot = mlp.forward(X_plot)

    plt.scatter(X_train, y_train, s=5, label='train', alpha=0.5)
    plt.plot(X_plot, y_plot, 'r-', lw=2, label='MLP_pre')
    plt.plot(X_plot, np.sin(X_plot), 'g--', lw=2, label='Real')
    plt.title('contrast', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # 开始预测
    prediction_menu(mlp)

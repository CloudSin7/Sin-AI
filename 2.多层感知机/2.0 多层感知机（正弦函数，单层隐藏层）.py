import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 强制使用 TkAgg 后端（pycharm问题）
import matplotlib.pyplot as plt


# 生成训练数据
def generate_data(num_samples):
    x = np.linspace(-2 * np.pi, 2 * np.pi, num_samples)
    y = np.sin(x) + np.random.randn(num_samples) * 0.1  # 添加少量噪声
    return x.reshape(-1, 1), y.reshape(-1, 1)


def display_data(x, y, n=5):
    print(f"生成数据集:")
    for i in range(n):
        print(f"X: {x[i]}, Y: {y[i]}")


# 定义多层感知机模型
class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重和偏置
        self.w1 = np.random.randn(input_size, hidden_size)  # 输入层到隐藏层的权重
        self.b1 = np.zeros((1, hidden_size))  # 隐藏层的偏置
        self.w2 = np.random.randn(hidden_size, output_size)  # 隐藏层到输出层的权重
        self.b2 = np.zeros((1, output_size))  # 输出层的偏置

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        # 前向传播
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        y_pred = self.sigmoid(self.z2)
        return y_pred

    def compute_loss(self, predictions, targets):
        # 计算损失（均方误差）
        return np.mean((predictions - targets) ** 2)

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        # 反向传播
        dz2 = 2 * (self.forward(X) - y) * (self.sigmoid(self.z2) * (1 - self.sigmoid(self.z2)))
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = np.dot(dz2, self.w2.T)
        dz1 = da1 * (self.sigmoid(self.z1) * (1 - self.sigmoid(self.z1)))
        dw1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # 更新参数
        self.w2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2
        self.w1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            predictions = self.forward(X)
            loss = self.compute_loss(predictions, y)
            self.backward(X, y, learning_rate)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")


# 用户交互菜单
def menu(model):
    while True:
        try:
            user_input = input("请输入一个数字或输入 'exit' 退出: ")
            if user_input.lower() == 'exit':
                break

            x_test = float(user_input)
            X_test = np.array([[x_test]])

            prediction = model.forward(X_test)
            print(f"预测结果: {prediction[0][0]}")
        except ValueError:
            print("输入格式不正确，请重新输入一个数字。")


# 主程序
if __name__ == "__main__":
    # 生成训练数据
    X_train, y_train = generate_data(1000)
    display_data(X_train, y_train)

    # 创建并训练模型
    mlp = SimpleMLP(input_size=1, hidden_size=70, output_size=1)
    mlp.train(X_train, y_train, learning_rate=0.0001, epochs=1000)

    # 进行预测并可视化结果
    X_test = np.linspace(-2 * np.pi, 2 * np.pi, 100).reshape(-1, 1)
    y_pred = mlp.forward(X_test)

    plt.scatter(X_train, y_train, label='Training Data', alpha=0.5)
    plt.plot(X_test, y_pred, color='red', label='Predicted Curve')
    plt.title('Nonlinear Regression with MLP')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    # 开始预测
    menu(mlp)

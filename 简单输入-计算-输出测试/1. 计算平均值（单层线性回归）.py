import numpy as np

# 生成训练数据
def generate_data(num_samples):
    x = []  # 数据
    y = []  # 目标值
    for _ in range(num_samples):
        # 随机生成两个数
        x1 = np.random.rand() * 10
        x2 = np.random.rand() * 10
        # 目标值为这两个数的平均值加上一个小的随机扰动
        target = (x1 + x2) / 2 + np.random.randn() * 0.1
        x.append([x1, x2])
        y.append(target)
    return np.array(x), np.array(y)


def display_data(x, y, n):
    print(f"生成数据集:")
    for i in range(n):
        print(f"X: {x[i]}, Y: {y[i]}")


# 因为我们设计的数据集是用平均数来弄的，所以这个数据集应该更符合线性回归的拟合，所以我们定义模型的时候就使用线性回归，
# 通过用直线去拟合输入得到输出
class SimpleLinearModel:
    def __init__(self):
        self.weights = np.random.randn(2)  # 初始化权重
        self.bias = np.random.randn()  # 初始化偏置

# 初始化模型的参数：权重（weights）和偏置（bias）。
# 权重：线性模型的核心参数，用于表示输入特征与输出之间的关系。因为我们有两个输入特征（x1 和 x2），所以权重是一个长度为2的向量。
# 偏置：一个常数项，用于调整模型的输出，使得模型更加灵活。它不依赖于输入特征。
# 随机初始化：生成随机值作为初始权重和偏置。随机初始化可以帮助模型在训练过程中探索不同的参数空间。
# 稍后要使用收入数据与权重进行线性计算，所以维度（数据个数）必须匹配

    def forward(self, x):
        """前向传播"""
        return np.dot(x, self.weights) + self.bias

# 前向传播过程，根据输入数据 x 计算预测值。
# 通过矩阵乘法 np.dot 将输入特征与权重进行线性组合，再加上偏置项 self.bias，得到预测值。
# 本质：输入特征为 x = [x1, x2]，则预测值 y_pred = x1 * w1 + x2 * w2 + b，其中 w1 和 w2 是权重，b 是偏置。
# 通过调整权重和偏置，模型可以在不同的输入特征之间找到合适的线性关系，实现预测

    def compute_loss(self, predictions, targets):
        """计算均方误差损失"""
        return np.mean((predictions - targets) ** 2)

# 计算模型预测值与实际目标值之间的差异，衡量模型的性能。
# 均方误差（MSE）：损失函数定义为预测值与目标值之差的平方的平均值。
# 这样一来，当误差较大的时候，给出的 loss 偏差增长更快，有利于模型快速拟合

    def train(self, x, y, learning_rate=0.01, epochs=1000):
        """训练模型"""
        for epoch in range(epochs):
            predictions = self.forward(x)
            loss = self.compute_loss(predictions, y)

            # 计算梯度
            gradients_w = 2 * np.dot(x.T, predictions - y) / len(y)
            gradients_b = 2 * np.mean(predictions - y)

            # 更新权重和偏置
            self.weights -= learning_rate * gradients_w
            self.bias -= learning_rate * gradients_b

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

# 通过多次迭代更新模型参数，使模型逐渐逼近最优解。
# 梯度下降：通过计算损失函数关于模型参数的梯度，并沿着梯度的反方向更新参数，以最小化损失函数。具体步骤如下：

# 前向传播：计算当前参数下的预测值。
# 计算损失：根据预测值和实际目标值计算损失。
# 计算梯度：使用链式法则计算损失函数关于权重和偏置的梯度。
# 更新参数：根据学习率调整参数

# 为何这样设计
# 迭代优化：梯度下降是一种常用的优化算法，能够有效地找到损失函数的局部最小值。
# 学习率控制：学习率决定的是每次更新的步长，选择合适的学习率对于模型收敛至关重要。过大的学习率可能导致模型无法收敛，而过小的学习率会导致收敛速度过慢。
# 批量更新：在这个实现中，使用的是批量梯度下降，即每次更新都基于整个数据集的梯度。这保证了每一步更新的方向是最优的，但可能需要更多的内存和计算资源。

# 梯度可以理解为损失函数的求导结果，表明了往哪里调整可以让损失函数的值更接近0，其实也就是我们的模型输出更接近结果值
# 梯度下降法的基本思想是沿着梯度的反方向更新参数，以减小损失函数的值。我们使用公式 新参数=旧参数-学习率*梯度来更新参数。
# 所以学习率其实就是每次调整的时候调整的程度的值，所以有学习率控制这一部分内容
def menu(model):
    while True:
        try:
            # 获取用户输入
            user_input = input("请输入两个数字（以空格分隔）或输入 'exit' 退出: ")
            if user_input.lower() == 'exit':
                break

            # 解析用户输入
            x1, x2 = map(float, user_input.split())
            X_test = np.array([[x1, x2]])

            # 进行预测
            prediction = model.forward(X_test)
            print(f"预测结果: {prediction[0]}")

        except ValueError:
            print("输入格式不正确，请重新输入两个数字。")


# 主程序
if __name__ == "__main__":
    # 生成训练数据
    X_train, y_train = generate_data(1000)

    # 显示前五行数据
    display_data(X_train, y_train, 5)

    # 创建并训练模型
    model = SimpleLinearModel()
    model.train(X_train, y_train)

    # 开始预测
    menu(model)

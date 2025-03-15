# -*- coding: utf-8 -*-
"""
MNIST手写识别 - 数据加载与模型定义模块
包含：数据加载、预处理、增强、MLP模型架构定义
"""

import numpy as np
import os
import struct
import gzip
import shutil
from scipy.ndimage import rotate
from sklearn.model_selection import train_test_split
from math import erf
import matplotlib
matplotlib.use('TkAgg')  # 强制使用 TkAgg 后端（pycharm问题）
import matplotlib.pyplot as plt

# ======================== 数据加载与预处理 ========================
class MNISTLoader:
    """MNIST数据加载与预处理专用类"""

    def __init__(self, data_dir="data/mnist"):
        self.data_dir = data_dir
        self.required_files = [
            "train-images-idx3-ubyte",
            "train-labels-idx1-ubyte",
            "t10k-images-idx3-ubyte",
            "t10k-labels-idx1-ubyte"
        ]
        self._prepare_data()

    def _prepare_data(self):
        """确保数据文件存在并已解压"""
        os.makedirs(self.data_dir, exist_ok=True)
        for file in self.required_files:
            self._decompress_file(file)

    def _decompress_file(self, filename):
        gz_path = os.path.join(self.data_dir, filename + ".gz")
        bin_path = os.path.join(self.data_dir, filename)

        if not os.path.exists(bin_path) and os.path.exists(gz_path):
            print(f"解压文件中: {filename}.gz")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(bin_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

    def load_data(self, augment=False):
        """加载并返回标准化后的数据集"""
        X_train = self._read_images("train-images-idx3-ubyte")
        y_train = self._read_labels("train-labels-idx1-ubyte")
        X_test = self._read_images("t10k-images-idx3-ubyte")
        y_test = self._read_labels("t10k-labels-idx1-ubyte")

        if augment:
            X_train, y_train = self._augment_data(X_train, y_train)

        # 划分验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42
        )
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def _read_images(self, filename):
        """读取图像数据并归一化到[0,1]"""
        with open(os.path.join(self.data_dir, filename), 'rb') as f:
            _, num, rows, cols = struct.unpack(">IIII", f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows*cols) / 255.0

    def _read_labels(self, filename):
        """读取标签数据"""
        with open(os.path.join(self.data_dir, filename), 'rb') as f:
            _, num = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    def _augment_data(self, images, labels, augment_factor=0.2):
        """数据增强：平移+旋转"""
        aug_images, aug_labels = [], []
        for img, label in zip(images, labels):
            aug_images.append(img)
            aug_labels.append(label)

            # 随机水平平移
            if np.random.rand() < augment_factor:
                dx = np.random.randint(-2, 3)
                shifted = np.roll(img.reshape(28,28), dx, axis=1).flatten()
                aug_images.append(shifted)
                aug_labels.append(label)

            # 随机旋转
            if np.random.rand() < augment_factor:
                angle = np.random.uniform(-15, 15)
                rotated = rotate(img.reshape(28,28), angle, reshape=False).flatten()
                aug_images.append(rotated)
                aug_labels.append(label)

        return np.array(aug_images), np.array(aug_labels)

# ======================== 模型核心定义 ========================
class Activation:
    """激活函数库（数值稳定版）"""

    @classmethod
    def get(cls, name):
        """获取激活函数"""
        return {
            'relu': cls.relu,
            'leaky_relu': cls.leaky_relu,
            'swish': cls.swish,
            'gelu': cls.gelu,
            'sigmoid': cls.sigmoid,
            'tanh': cls.tanh
        }[name.lower()]

    @classmethod
    def derivative(cls, name, a):
        """获取激活函数导数"""
        return {
            'relu': cls.relu_deriv,
            'leaky_relu': cls.leaky_relu_deriv,
            'swish': cls.swish_deriv,
            'gelu': cls.gelu_deriv,
            'sigmoid': cls.sigmoid_deriv,
            'tanh': cls.tanh_deriv
        }[name.lower()](a)

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_deriv(a):
        return (a > 0).astype(float)

    @staticmethod
    def leaky_relu(z, alpha=0.01):
        return np.where(z >= 0, z, alpha*z)

    @staticmethod
    def leaky_relu_deriv(a, alpha=0.01):
        return np.where(a > 0, 1, alpha)

    @staticmethod
    def swish(z):
        z_clip = np.clip(z, -50, 50)
        return z * (1 / (1 + np.exp(-z_clip)))

    @staticmethod
    def swish_deriv(a):
        sigmoid = 1 / (1 + np.exp(-np.clip(a, -50, 50)))
        return sigmoid + a * sigmoid * (1 - sigmoid)

    @staticmethod
    def gelu(z):
        return 0.5 * z * (1 + np.vectorize(erf)(z / np.sqrt(2)))

    @staticmethod
    def gelu_deriv(a):
        return 0.5 * (1 + np.vectorize(erf)(a / np.sqrt(2))) + \
            (0.5 * a * np.exp(-a**2/2)) / np.sqrt(np.pi/2)

    @staticmethod
    def sigmoid(z):
        z_clip = np.clip(z, -50, 50)
        return 1 / (1 + np.exp(-z_clip))

    @staticmethod
    def sigmoid_deriv(a):
        return a * (1 - a)

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def tanh_deriv(a):
        return 1 - a**2

class DynamicMLP:
    """可配置的增强型MLP"""

    def __init__(self, layer_sizes, activation='relu',
                 use_batchnorm=False, dropout_rate=0.2):
        """
        参数：
            layer_sizes : 网络结构列表，如[784, 256, 10]
            activation : 隐藏层激活函数（'relu'/'swish'等）
            use_batchnorm : 是否使用批量归一化
            dropout_rate : Dropout概率（0表示禁用）
        """
        self.layer_sizes = layer_sizes
        self.activation = activation.lower()
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate

        # 初始化参数
        self.weights, self.biases = [], []
        self.bn_gamma, self.bn_beta = [], []
        self._init_parameters()

        # 训练状态缓存
        self.cache = {}

    def _init_parameters(self):
        """参数初始化"""
        for i in range(len(self.layer_sizes)-1):
            in_size = self.layer_sizes[i]
            out_size = self.layer_sizes[i+1]

            # 权重初始化
            if self.activation in ['relu', 'leaky_relu']:
                # He初始化
                std = np.sqrt(2.0 / in_size)
            else:
                # Xavier初始化
                std = np.sqrt(1.0 / in_size)

            self.weights.append(np.random.randn(in_size, out_size) * std)
            self.biases.append(np.zeros((1, out_size)))

            # 批量归一化参数
            if self.use_batchnorm and i < len(self.layer_sizes)-2:
                self.bn_gamma.append(np.ones((1, out_size)))
                self.bn_beta.append(np.zeros((1, out_size)))

    def forward(self, X, training=True):
        """前向传播（数值稳定版）"""
        self.cache = {'a': [X], 'z': [], 'mask': [], 'bn': []}
        a = X

        for i in range(len(self.weights)):
            # 线性变换
            z = np.dot(a, self.weights[i]) + self.biases[i]

            # 批量归一化
            if self.use_batchnorm and i < len(self.weights)-1:
                if training:
                    mu, var = np.mean(z, axis=0), np.var(z, axis=0)
                else:
                    mu, var = self.running_mu[i], self.running_var[i]

                z_norm = (z - mu) / np.sqrt(var + 1e-5)
                z = self.bn_gamma[i] * z_norm + self.bn_beta[i]
                self.cache['bn'].append((mu, var, z_norm))

            # 激活函数（输出层除外）
            if i < len(self.weights)-1:
                a = Activation.get(self.activation)(z)
                a = np.clip(a, 1e-8, None)  # 防止数值下溢

                # Dropout
                if training and self.dropout_rate > 0:
                    mask = (np.random.rand(*a.shape) > self.dropout_rate)
                    a *= mask / (1 - self.dropout_rate)
                    self.cache['mask'].append(mask)
            else:
                a = self._stable_softmax(z)

            self.cache['z'].append(z)
            self.cache['a'].append(a)

        return a

    def _stable_softmax(self, z):
        """数值稳定的Softmax"""
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-8)

# ======================== 训练流程控制 ========================
class MLPTrainer:
    """模块化训练控制器"""

    def __init__(self, model, patience=5, lr_scheduler='cosine'):
        """
        参数：
            model : 要训练的MLP模型实例
            patience : 早停耐心值（连续不提升的epoch数）
            lr_scheduler : 学习率调度策略（'cosine'/'step'/'none'）
        """
        self.model = model
        self.patience = patience
        self.lr_scheduler = lr_scheduler
        self.best_loss = np.inf
        self.no_improve = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

    def train(self, train_data, val_data,
              epochs=50, batch_size=64,
              init_lr=0.001, l2_lambda=0.0001,
              grad_clip=5.0):
        """
        执行完整训练流程
        参数：
            train_data : (X_train, y_train) 训练数据
            val_data : (X_val, y_val) 验证数据
            epochs : 最大训练轮数
            batch_size : 批次大小
            init_lr : 初始学习率
            l2_lambda : L2正则化系数
            grad_clip : 梯度裁剪阈值
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        self.epochs = epochs
        self.base_lr = init_lr

        for epoch in range(epochs):
            # 学习率调整
            current_lr = self._get_learning_rate(epoch)
            self.history['lr'].append(current_lr)

            # 训练单个epoch
            train_loss = self._train_epoch(
                X_train, y_train,
                batch_size, current_lr,
                l2_lambda, grad_clip
            )

            # 验证评估
            val_loss, val_acc = self._evaluate(X_val, y_val)

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # 打印进度
            log = (f"Epoch {epoch+1}/{epochs} | "
                   f"LR: {current_lr:.5f} | "
                   f"Train Loss: {train_loss:.4f} | "
                   f"Val Loss: {val_loss:.4f} | "
                   f"Val Acc: {val_acc:.4f}")
            print(log)

            # 早停检查
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.no_improve = 0
                self._save_best_weights()
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    print(f"早停触发，在第 {epoch+1} 轮停止训练")
                    self._restore_best_weights()
                    break

        return self.history

    def _train_epoch(self, X, y, batch_size, lr, l2_lambda, grad_clip):
        """单epoch训练"""
        total_loss = 0
        indices = np.random.permutation(len(X))
        X_shuffled, y_shuffled = X[indices], y[indices]

        for i in range(0, len(X), batch_size):
            # 获取当前批次
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # 前向传播
            probs = self.model.forward(X_batch, training=True)

            # 计算损失（数值稳定）
            probs = np.clip(probs, 1e-8, 1-1e-8)
            batch_loss = -np.mean(np.log(probs[np.arange(len(y_batch)), y_batch]))
            total_loss += batch_loss * len(y_batch)

            # 反向传播
            self._backward_step(X_batch, y_batch, lr,
                                l2_lambda, grad_clip)

        return total_loss / len(X)

    def _backward_step(self, X_batch, y_batch, lr, l2_lambda, grad_clip):
        """执行反向传播并更新参数"""
        # 清空旧梯度
        gradients_w = [np.zeros_like(w) for w in self.model.weights]
        gradients_b = [np.zeros_like(b) for b in self.model.biases]

        # 计算输出层梯度
        probs = self.model.cache['a'][-1]
        delta = probs.copy()
        delta[np.arange(len(y_batch)), y_batch] -= 1
        delta /= len(y_batch)


    def _backward_step(self, X_batch, y_batch, lr, l2_lambda, grad_clip):
        """反向传播方法"""
        gradients_w = [np.zeros_like(w) for w in self.model.weights]
        gradients_b = [np.zeros_like(b) for b in self.model.biases]

        # 初始化输出层梯度
        probs = self.model.cache['a'][-1]
        delta = probs.copy()
        delta[np.arange(len(y_batch)), y_batch] -= 1
        delta /= len(y_batch)

        # 反向遍历所有层（从输出层到输入层）
        for layer_idx in reversed(range(len(self.model.weights))):
            # 获取当前层输入激活值
            a_prev = self.model.cache['a'][layer_idx]  # layer_idx对应当前层

            # 计算当前层梯度
            gradients_w[layer_idx] = np.dot(a_prev.T, delta) + l2_lambda * self.model.weights[layer_idx]
            gradients_b[layer_idx] = np.sum(delta, axis=0, keepdims=True)

            # 梯度裁剪
            gradients_w[layer_idx] = np.clip(gradients_w[layer_idx], -grad_clip, grad_clip)
            gradients_b[layer_idx] = np.clip(gradients_b[layer_idx], -grad_clip, grad_clip)

            # 计算前一层delta（输入层不需要）
            if layer_idx > 0:
                # 传播误差
                delta = np.dot(delta, self.model.weights[layer_idx].T)

                # 应用激活函数导数
                z = self.model.cache['z'][layer_idx-1]  # z对应前一层
                delta *= Activation.derivative(self.model.activation,
                                               self.model.cache['a'][layer_idx])

                # 应用Dropout mask
                if self.model.dropout_rate > 0 and layer_idx < len(self.model.weights)-1:
                    mask = self.model.cache['mask'][layer_idx-1]
                    delta *= mask

                # 应用BatchNorm梯度
                if self.model.use_batchnorm and layer_idx > 0:
                    _, var, z_norm = self.model.cache['bn'][layer_idx-1]
                    delta = (self.model.bn_gamma[layer_idx-1] / np.sqrt(var + 1e-5)) * delta

        # 更新参数
        for layer_idx in range(len(self.model.weights)):
            self.model.weights[layer_idx] -= lr * gradients_w[layer_idx]
            self.model.biases[layer_idx] -= lr * gradients_b[layer_idx]
    def _evaluate(self, X, y):
        """在验证集/测试集上评估"""
        probs = self.model.forward(X, training=False)
        probs = np.clip(probs, 1e-8, 1-1e-8)
        loss = -np.mean(np.log(probs[np.arange(len(y)), y]))
        preds = np.argmax(probs, axis=1)
        acc = np.mean(preds == y)
        return loss, acc

    def _get_learning_rate(self, epoch):
        """学习率调度"""
        if self.lr_scheduler == 'cosine':
            return self.base_lr * 0.5 * (1 + np.cos(np.pi * epoch / self.epochs))
        elif self.lr_scheduler == 'step':
            return self.base_lr * (0.1 ** (epoch // 30))
        else:
            return self.base_lr

    def _save_best_weights(self):
        """保存最佳参数"""
        self.best_weights = [w.copy() for w in self.model.weights]
        self.best_biases = [b.copy() for b in self.model.biases]
        if self.model.use_batchnorm:
            self.best_bn_gamma = [g.copy() for g in self.model.bn_gamma]
            self.best_bn_beta = [bt.copy() for bt in self.model.bn_beta]

    def _restore_best_weights(self):
        """恢复最佳参数"""
        self.model.weights = [w.copy() for w in self.best_weights]
        self.model.biases = [b.copy() for b in self.best_biases]
        if self.model.use_batchnorm:
            self.model.bn_gamma = [g.copy() for g in self.best_bn_gamma]
            self.model.bn_beta = [bt.copy() for bt in self.best_bn_beta]
# ======================== 评估与可视化 ========================
class ModelEvaluator:
    """模型性能评估与可视化工具集"""

    @classmethod
    def generate_classification_report(cls, model, X_test, y_test):
        """
        生成详细分类报告
        参数：
            model : 训练好的模型实例
            X_test : 测试集特征
            y_test : 测试集真实标签
        返回：
            pandas DataFrame格式的分类报告
        """
        from sklearn.metrics import classification_report
        import pandas as pd

        # 获取预测结果
        probs = model.forward(X_test, training=False)
        y_pred = np.argmax(probs, axis=1)

        # 生成报告
        report = classification_report(
            y_test, y_pred,
            target_names=[str(i) for i in range(10)],
            output_dict=True
        )

        # 转换为DataFrame并美化输出
        df_report = pd.DataFrame(report).transpose()
        df_report['support'] = df_report['support'].astype(int)
        return df_report.round(4)

    @classmethod
    def plot_confusion_matrix(cls, model, X_test, y_test, figsize=(12,10)):
        """
        绘制混淆矩阵热力图
        参数：
            model : 训练好的模型实例
            X_test : 测试集特征
            y_test : 测试集真实标签
            figsize : 图像尺寸
        """
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        # 获取预测结果
        probs = model.forward(X_test, training=False)
        y_pred = np.argmax(probs, axis=1)

        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)

        # 绘制热力图
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    @classmethod
    def visualize_feature_space(cls, model, X, y, sample_size=1000, figsize=(12,8)):
        """
        可视化隐藏层特征空间（PCA降维）
        参数：
            model : 训练好的模型实例
            X : 输入数据
            y : 真实标签
            sample_size : 采样数量（大数据集时加速可视化）
            figsize : 图像尺寸
        """
        from sklearn.decomposition import PCA
        import matplotlib.colors as mcolors

        # 随机采样
        indices = np.random.choice(len(X), size=sample_size, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]

        # 提取最后一个隐藏层激活值
        _ = model.forward(X_sample, training=False)
        hidden_activation = model.cache['a'][-2]  # 倒数第二层为最后一个隐藏层

        # PCA降维到2D
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(hidden_activation)

        # 创建颜色映射
        cmap = plt.get_cmap('tab10')
        norm = mcolors.Normalize(vmin=0, vmax=9)

        # 绘制散点图
        plt.figure(figsize=figsize)
        scatter = plt.scatter(
            features_2d[:,0], features_2d[:,1],
            c=y_sample, cmap=cmap, norm=norm,
            alpha=0.6, edgecolors='w', linewidth=0.5
        )

        # 添加图例
        cbar = plt.colorbar(scatter, ticks=range(10))
        cbar.set_label('Digit Class')
        plt.title('Hidden Layer Feature Space (PCA Projection)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()

    @classmethod
    def plot_activation_distribution(cls, model, X_sample):
        """
        绘制各层激活值分布直方图
        参数：
            model : 训练好的模型实例
            X_sample : 用于分析的输入样本
        """
        # 前向传播获取各层激活值
        _ = model.forward(X_sample, training=False)

        # 创建子图
        plt.figure(figsize=(15, 4*len(model.cache['a'])))
        for i, a in enumerate(model.cache['a'][1:-1]):  # 排除输入层和输出层
            plt.subplot(len(model.cache['a'])-2, 1, i+1)
            plt.hist(a.flatten(), bins=50, alpha=0.7)
            plt.title(f'Layer {i+1} Activation Distribution')
            plt.xlabel('Activation Value')
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_weight_distribution(cls, model, figsize=(15, 10)):
        """
        绘制各层权重分布直方图
        参数：
            model : 训练好的模型实例
            figsize : 图像尺寸
        """
        plt.figure(figsize=figsize)
        for i, w in enumerate(model.weights):
            plt.subplot(len(model.weights), 1, i+1)
            plt.hist(w.flatten(), bins=50, alpha=0.7)
            plt.title(f'Layer {i+1} Weight Distribution')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    @classmethod
    def visualize_misclassified(cls, model, X_test, y_test, num_samples=25):
        """
        可视化错误分类样本
        参数：
            model : 训练好的模型实例
            X_test : 测试集特征
            y_test : 测试集真实标签
            num_samples : 显示样本数量
        """
        # 获取预测结果
        probs = model.forward(X_test, training=False)
        y_pred = np.argmax(probs, axis=1)

        # 找出错误分类索引
        wrong_indices = np.where(y_pred != y_test)[0]
        if len(wrong_indices) == 0:
            print("没有错误分类样本！")
            return

        # 随机选择部分样本
        selected = np.random.choice(wrong_indices, size=min(num_samples, len(wrong_indices)), replace=False)

        # 绘制图像
        plt.figure(figsize=(10, 10))
        for i, idx in enumerate(selected):
            plt.subplot(5, 5, i+1)
            img = X_test[idx].reshape(28, 28)
            plt.imshow(img, cmap='gray')
            plt.title(f"True: {y_test[idx]}\nPred: {y_pred[idx]}", fontsize=8)
            plt.axis('off')
        plt.tight_layout()
        plt.show()
# ======================== 模型保存 ========================
def save_model(model, filepath):
    """简化版模型保存，仅保存必要参数"""
    import pickle
    import datetime

    params = {
        'weights': [w.copy() for w in model.weights],
        'biases': [b.copy() for b in model.biases],
        'config': {
            'layer_sizes': model.layer_sizes,
            'activation': model.activation,
            'use_batchnorm': model.use_batchnorm,
            'dropout_rate': model.dropout_rate
        },
        'save_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(filepath, 'wb') as f:
        pickle.dump(params, f)
    print(f"模型已成功保存至：{filepath}")

# ======================== 主程序 ========================
def main():
    print("\n" + "="*50)
    print("=== MNIST手写数字识别训练系统 ===")
    print("="*50)

    # 用户参数设置
    print("\n【模型结构配置】")
    hidden_units = input("请输入隐藏层神经元数量（逗号分隔，推荐 128,64）：") or "128,64"
    layer_sizes = [784] + list(map(int, hidden_units.split(','))) + [10]

    print("\n【激活函数选择】")
    print("可选选项：1.ReLU（默认） 2.Swish 3.LeakyReLU")
    activation_choice = input("请选择激活函数（输入数字）：") or "1"
    activations = {'1':'relu', '2':'swish', '3':'leaky_relu'}
    activation = activations.get(activation_choice, 'relu')

    print("\n【训练参数设置】")
    epochs = int(input(f"训练轮数（推荐 30-100，默认50）：") or 50)
    batch_size = int(input(f"批大小（推荐 64-256，默认128）：") or 128)
    learning_rate = float(input(f"学习率（推荐 0.001-0.1，默认0.01）：") or 0.01)

    # 自动配置推荐参数
    print("\n【自动配置项】")
    print("- 输出层激活函数：Softmax（自动设置）")
    print("- 使用批量归一化：是（推荐）")
    print("- Dropout比例：0.3（推荐）")

    # 初始化模型
    model = DynamicMLP(
        layer_sizes=layer_sizes,
        activation=activation,
        use_batchnorm=True,  # 强制启用批量归一化
        dropout_rate=0.3      # 固定Dropout比例
    )

    # 加载数据
    print("\n【数据加载中...】")
    loader = MNISTLoader()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.load_data(augment=True)

    # 初始化训练器
    trainer = MLPTrainer(model, patience=5, lr_scheduler='cosine')

    # 开始训练
    print("\n【开始训练】")
    history = trainer.train(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        init_lr=learning_rate,
        l2_lambda=0.0001,
        grad_clip=5.0
    )

    # 训练结果展示
    print("\n【训练结果摘要】")
    final_acc = history['val_acc'][-1]
    print(f"- 最终验证准确率：{final_acc:.2%}")
    print(f"- 最佳验证准确率：{max(history['val_acc']):.2%}")

    # 随机测试展示
    print("\n【随机测试展示】")
    num_samples = 5
    sample_indices = np.random.choice(len(X_test), num_samples)
    samples = X_test[sample_indices]
    labels = y_test[sample_indices]

    # 进行预测
    probs = model.forward(samples, training=False)
    preds = np.argmax(probs, axis=1)

    # 展示结果
    plt.figure(figsize=(12, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(samples[i].reshape(28,28), cmap='gray')
        plt.title(f"真实: {labels[i]}\n预测: {preds[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 模型保存选项
    save_choice = input("\n是否保存模型？(y/n): ").lower()
    if save_choice == 'y':
        model_name = input("请输入模型名称（无需后缀）：") or "mnist_model"
        save_model(model, f"{model_name}.pkl")
        print("保存成功！")
    else:
        print("未保存模型")

    print("\n=== 训练流程结束 ===")

if __name__ == "__main__":
    main()
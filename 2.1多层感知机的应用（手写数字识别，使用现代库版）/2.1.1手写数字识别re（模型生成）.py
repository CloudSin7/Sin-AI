import datetime
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, optimizers, initializers
import matplotlib
matplotlib.use('TkAgg')  # 强制使用 TkAgg 后端（pycharm问题）
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import struct
import gzip
import shutil
import glob


# ------------------------- 数据加载 -------------------------
def load_mnist(data_dir="data/mnist"):
    def read_images(path):
        with open(path, 'rb') as f:
            _, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols) / 255.0
            return images

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
    print("\n" + "=" * 40)
    print("=== MNIST MLP 参数设置 ===")
    params = {'layers': [784] + list(map(int, (input("隐藏层结构（逗号分隔，如128,64）: ") or "128").split(','))) + [10],
              'epochs': int(input("训练轮数 (默认20): ") or 20),
              'batch_size': int(input("批量大小 (默认64): ") or 64),
              'learning_rate': float(input("学习率 (默认0.001): ") or 0.001),
              'activation': input("激活函数（relu/sigmoid/tanh，默认relu）: ").lower() or 'relu',
              'use_early_stopping': input("是否启用早停机制 (yes/no，默认yes): ").lower() or 'yes'
              }
    return params


# ------------------------- 模型构建 -------------------------
def build_model(params):
    # 初始化策略
    init = initializers.HeNormal() if params['activation'] == 'relu' \
        else initializers.GlorotNormal()

    model = tf.keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(params['layers'][1],
                     activation=params['activation'],
                     kernel_initializer=init),
        layers.BatchNormalization(),  # 加入 Batch Normalization
        layers.Dropout(0.5)  # 加入 Dropout
    ])

    # 添加隐藏层
    for units in params['layers'][2:-1]:
        model.add(layers.Dense(units,
                               activation=params['activation'],
                               kernel_initializer=init))
        model.add(layers.BatchNormalization())  # 加入 Batch Normalization
        model.add(layers.Dropout(0.5))  # 加入 Dropout


# 输出层
    model.add(layers.Dense(10, activation='softmax'))

    # 使用 Adam 优化器
    model.compile(optimizer=optimizers.Adam(learning_rate=params['learning_rate']),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# ------------------------- 训练可视化 -------------------------
def visualize_training(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.title('训练损失曲线')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('准确率曲线')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------------- 模型展示与评估 -------------------------
def show_model_params(model):
    print(f"\n{'=' * 50}\n=== 模型结构 ===")
    for i, layer in enumerate(model.layers):
        config = layer.get_config()
        input_shape = config.get('batch_input_shape', 'Unknown')
        units = config.get('units', 'Unknown')
        print(f"层 {i + 1} (类型: {layer.__class__.__name__}, 输入形状: {input_shape}, 输出单元: {units})")
        weights = layer.get_weights()
        if weights:
            w = weights[0]
            print(f"  权重范围: [{w.min():.4f}, {w.max():.4f}]")
            if len(weights) > 1:  # 检查是否有偏置
                b = weights[1]
                print(f"  偏置范围: [{b.min():.4f}, {b.max():.4f}]")
    print("=" * 50)

from sklearn.metrics import confusion_matrix
import seaborn as sns


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred_classes)

    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # 计算每个类别的准确率
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(class_accuracy):
        print(f"Class {i} Accuracy: {acc:.4f}")


# ------------------------- 保存模型 -------------------------
def generate_model_name(params, test_acc):
    hidden_layers = '+'.join(map(str, params['layers'][1:-1]))  # 隐藏层结构
    epochs = params['epochs']  # 训练轮次
    batch_size = params['batch_size']  # 批量大小
    activation = params['activation']  # 激活函数
    accuracy = f"{test_acc:.5f}"  # 最终准确率
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")  # 月日时分时间戳
    return f"{hidden_layers}_{epochs}_{batch_size}_{activation}_{accuracy}_{timestamp}.keras"

def save_model(model, params, test_acc, save_dir="modules"):
    os.makedirs(save_dir, exist_ok=True)
    model_name = generate_model_name(params, test_acc)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print(f"模型已保存至: {model_path}")

# ------------------------- 主程序 -------------------------
if __name__ == "__main__":
    # 加载数据
    X_train, y_train, X_test, y_test = load_mnist()

    # 用户参数
    params = get_user_input()

    # 构建模型
    model = build_model(params)
    model.summary()  # 显示模型结构
    show_model_params(model)

    from tensorflow.keras.callbacks import EarlyStopping

    # 训练部分：
    callbacks = []
    if params['use_early_stopping'] == 'yes':
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        callbacks.append(early_stopping)

    history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=params['batch_size'],
            epochs=params['epochs'],
            verbose=1,
            callbacks=callbacks
        )

    # 训练过程可视化
    visualize_training(history)
    # 最终评估
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n测试集准确率: {test_acc:.4f}")
    evaluate_model(model, X_test, y_test)

    # 保存模型
    is_save = input("保存到本地(yes/no): ")
    if is_save.lower() == "yes":
        save_model(model, params, test_acc)
    else:
        print("程序结束")
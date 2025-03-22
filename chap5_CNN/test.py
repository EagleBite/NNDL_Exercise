import tensorflow as tf
from tqdm import tqdm

# 打印 TensorFlow 版本
print(tf.__version__)

# 加载 MNIST 数据集
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 归一化数据并调整形状
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images.reshape(-1, 28, 28, 1)  # 添加通道维度
test_images = test_images.reshape(-1, 28, 28, 1)    # 添加通道维度

# 将标签转换为 one-hot 编码
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# 定义模型
model = tf.keras.Sequential([
    # 卷积层 1
    tf.keras.layers.Conv2D(32, (7, 7), activation='relu', padding='same', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='same'),

    # 卷积层 2
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='same'),

    # 全连接层 1
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.3),  # Dropout 层

    # 全连接层 2（输出层）
    tf.keras.layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
max_epoch = 20  # 减少 epoch 数量以加快训练
batch_size = 100

# 使用 tqdm 显示训练进度
for epoch in range(max_epoch):
    print(f"Epoch {epoch + 1}/{max_epoch}")
    for i in tqdm(range(0, len(train_images), batch_size)):
        batch_images = train_images[i:i + batch_size]
        batch_labels = train_labels[i:i + batch_size]
        model.train_on_batch(batch_images, batch_labels)

    # 每个 epoch 结束后评估模型
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")

# 最终评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print(f"Final Test Accuracy: {test_acc:.4f}")
# 无监督学习:CWOLa


# 导入必要的库
from sklearn.metrics import roc_curve, auc
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf

# 数据准备
# 无监督学习的模型训练无需用到真实标签，而是对两个混合样本中的元素分别标记成0和1，制作成样本标签
# 训练集：x_train_1和x_train_2指两个不同信号比例的混合波形样本;
# 测试集：x_test_1,x_test_2同上
x_train_mixed = np.concatenate([x_train_1, x_train_2], axis=0)
y_train_mixed = np.concatenate([np.zeros(x_train_1.shape[0]), np.ones(x_train_2.shape[0])], axis=0)
x_test_mixed = np.concatenate([x_test_1, x_test_2], axis=0)
y_test_mixed = np.concatenate([np.zeros(x_test_1.shape[0]), np.ones(x_test_2.shape[0])], axis=0)
print(x_train_mixed)
print(x_train_mixed.shape)
print(y_train_mixed)
print(x_test_mixed)
print(x_test_mixed.shape)
print(y_test_mixed)

# 制作训练和测试的二分类标签
num_classes = 2
input_shape = (50,10,1)
y_train_mixed = keras.utils.to_categorical(y_train_mixed, num_classes)
y_test_mixed = keras.utils.to_categorical(y_test_mixed, num_classes)
print(y_train_mixed)
print(y_test_mixed)

# 简单搭建一个基于keras和tesorflow平台的二维卷积网络模型，编译并训练模型
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adam")
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

data_augmentation = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                                                                    width_shift_range=0.2,
                                                                    height_shift_range=0.2,
                                                                    shear_range=0.15,
                                                                    horizontal_flip=True,
                                                                    fill_mode="nearest")
conv_hist = model.fit(data_augmentation.flow(x_train_mixed, y_train_mixed, batch_size=128),
                    epochs=30, validation_data=(x_test_mixed, y_test_mixed))

# 训练过程图看是否正常
plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.title("Convolution Loss")
plt.plot(conv_hist.history["loss"], label="loss")
plt.plot(conv_hist.history["val_loss"], label="val_loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.title("Convolution Accuracy")
plt.plot(conv_hist.history["accuracy"], label="accuracy")
plt.plot(conv_hist.history["val_accuracy"], label="val_accuracy")
plt.legend()
plt.show()

# 确定类别
# x_test_mixed是模型训练过程中的验证集，无监督学习模式需要根据预测概率值画出直方图，一般选取尖峰之间的谷值作为阈值来区分类别
samples_predictions = model.predict(x_test_mixed)
samples_predictions_data = samples_predictions[:,0]
print(samples_predictions_data)
plt.hist(samples_predictions_data,bins=100,edgecolor="r",histtype="bar",alpha=0.5,label="threshold = 0.635")
plt.xlabel("Prediction Probability")
plt.ylabel("Sample Count")
plt.legend()
plt.show()

# x_test_1是测试集
signal_predictions = model.predict(x_test_1)
signal_predictions_data = signal_predictions[:,0]
signal_threshold = 0.635        # 选取直方图尖峰之间的谷值作为阈值来区分类别
for i in range(len(signal_predictions_data)):
    if signal_predictions_data[i] < signal_threshold:
        signal_predictions_data[i] = 1
    else:
        signal_predictions_data[i] = 0

# 评估性能
# y_test_1是测试集波形对应的真实光电子数，用于检测分类模型分类准确率及错误类型
accuracy = np.sum(signal_predictions_data == y_test_1) / len(y_test_1)
print('分类准确率:', accuracy)
cm = confusion_matrix(y_test_1, signal_predictions_data)
tn, fp, fn, tp = cm.ravel()
noise_acc = tn / (tn + fp)
signal_acc = tp / (tp + fn)
print('噪声准确率:', noise_acc)
print('信号准确率:', signal_acc)
print(cm)

# 画ROC曲线进一步验证模型性能的优异
fpr, tpr, thresholds = roc_curve(y_test_1, signal_predictions_data)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='(AUC = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()

# 如果想实现多标签分类，需要修改部分代码
# 如下三标签分类部分代码修改
num_classes = 2
# 由于无监督学习模式是采用训练区分两个混合样本的分类模型的输出来画直方图来确定阈值来区分三类别，并不需要更改模型的分类数量
signal_threshold = 0.650
mult_signal_threshold = 0.480   # 选取直方图尖峰之间的谷值作为阈值来区分类别，三分类模型需要选取两个阈值
for i in range(len(signal_predictions_data)):
    if signal_predictions_data[i] < mult_signal_threshold:
        signal_predictions_data[i] = 2
    elif signal_predictions_data[i] > signal_threshold:
        signal_predictions_data[i] = 0
    else:
        signal_predictions_data[i] = 1
accuracy = np.sum(signal_predictions_data == y_test_1) / len(y_test_1)
print('三标签分类的分类准确率:', accuracy)
cm = confusion_matrix(y_test_1, signal_predictions_data)
noise_acc = cm[0, 0] / np.sum(cm[0])
signal_acc = cm[1, 1] / np.sum(cm[1])
multi_signal_acc = cm[2, 2] / np.sum(cm[2])
print('Noise accuracy:', noise_acc)
print('Signal accuracy:', signal_acc)
print('Multi-signal accuracy:', multi_signal_acc)
print(cm)

# 三分类模型无法直接画出ROC曲线，但是可以利用macro算法来画出平均ROC曲线
signal_predictions_data = keras.utils.to_categorical(signal_predictions_data, num_classes)
y_test_1 = keras.utils.to_categorical(y_test_1, num_classes)
fpr_macro = []
tpr_macro = []
roc_auc_macro = []
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_test_1[:,i], signal_predictions_data[:,i])
    roc_auc = auc(fpr, tpr)
    fpr_macro.append(fpr)
    tpr_macro.append(tpr)
    roc_auc_macro.append(roc_auc)
fpr1 = np.average(fpr_macro, axis=0)
tpr1 = np.average(tpr_macro, axis=0)
roc_auc1 = np.average(roc_auc_macro, weights=[1/num_classes]*num_classes)
plt.figure()
plt.plot(fpr1, tpr1, color='blue', lw=2, label='(AUC = %0.4f)' % roc_auc1)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()



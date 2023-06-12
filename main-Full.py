# 完全监督学习:Full


# 导入必要的库
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 数据准备
# 训练集：x_train_1和x_train_2指两个不同信号比例的混合波形样本；y_train_1和y_train_2指两个混合波形样本的真实标签
# 测试集：x_test_1,x_test_2,y_test_1和y_test_2同上
x_train_mixed = np.concatenate([x_train_1, x_train_2], axis=0)
y_train_mixed = np.concatenate([y_train_1, y_train_2], axis=0)
x_test_mixed = np.concatenate([x_test_1, x_test_2], axis=0)
y_test_mixed = np.concatenate([y_test_1, y_test_2], axis=0)
print(x_train_mixed)
print(y_train_mixed)
print(x_test_mixed)
print(y_test_mixed)

# 制作训练和测试的二分类标签
num_classes = 2
input_shape = (50,10,1)
y_train_new = keras.utils.to_categorical(y_train_mixed, num_classes)
y_test_new = keras.utils.to_categorical(y_test_mixed, num_classes)
print(y_train_new)
print(y_test_new)

# 简单搭建一个基于keras和tesorflow平台的二维卷积网络模型，编译并训练模型
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
conv_hist = model.fit(x_train_mixed, y_train_mixed, batch_size=128, epochs=30, validation_data=(x_test_mixed, y_test_mixed))

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
# x_test_1是测试集，完全监督模式根据预测概率值最大来确定类别
signal_predictions = model.predict(x_test_1)
predictions = np.argmax(signal_predictions, axis=1)
for i in range(predictions.shape[0]):
    if predictions[i] == 0:
        signal_predictions[i] = [0, 1]
    elif predictions[i] == 1:
        signal_predictions[i] = [1, 0]
print(signal_predictions)
signal_predictions_data = signal_predictions[:,0]

# 评估性能
# y_test_1是测试集波形对应的真实光电子数，用于检测分类模型分类准确率及错误类型
accuracy = np.sum(signal_predictions_data == y_test_1) / len(y_test_1)
print('分类准确率:', accuracy)
cm = confusion_matrix(y_test_1, signal_predictions_data)
noise_acc = cm[0, 0] / np.sum(cm[0])
signal_acc = cm[1, 1] / np.sum(cm[1])
print('Noise accuracy:', noise_acc)
print('Signal accuracy:', signal_acc)
print(cm)

# 画ROC曲线进一步验证模型性能的优异
signal_predictions_data = signal_predictions[:,0]
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
# 模型训练分成几类别
num_classes = 3
# 根据预测概率值最大来确定三个类别
signal_predictions = model.predict(x_test_1)
predictions = np.argmax(signal_predictions, axis=1)
for i in range(predictions.shape[0]):
    if predictions[i] == 0:
        signal_predictions[i] = [0, 0, 0]
    elif predictions[i] == 1:
        signal_predictions[i] = [1, 1, 1]
    elif predictions[i] == 2:
        signal_predictions[i] = [2, 2, 2]
print(signal_predictions)
signal_predictions_data = signal_predictions[:,0]
accuracy = np.sum(signal_predictions_data == y_test_1) / len(y_test_1)
print('三标签分类的分类准确率:', accuracy)
cm = confusion_matrix( y_test_1, signal_predictions_data)
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


# coding: utf-8



from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('/data/dog_images/train')
valid_files, valid_targets = load_dataset('/data/dog_images/valid')
test_files, test_targets = load_dataset('/data/dog_images/test')

# load list of dog names
dog_names = [item[27:-1] for item in sorted(glob("/data/dog_images/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))



import random
random.seed(8675309)

# 加载打乱后的人脸数据集的文件名
human_files = np.array(glob("/data/lfw/*/*"))
random.shuffle(human_files)

# 打印数据集的数据量
print('There are %d total human images.' % len(human_files))





import cv2                
import matplotlib.pyplot as plt                        
get_ipython().run_line_magic('matplotlib', 'inline')

# 提取预训练的人脸检测模型
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# 加载彩色（通道顺序为BGR）图像
img = cv2.imread(human_files[3])

# 将BGR图像进行灰度处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 在图像中找出脸
faces = face_cascade.detectMultiScale(gray)

# 打印图像中检测到的脸的个数
print('Number of faces detected:', len(faces))

# 获取每一个所检测到的脸的识别框
for (x,y,w,h) in faces:
    # 在人脸图像中绘制出识别框
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# 将BGR图像转变为RGB图像以打印
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 展示含有识别框的图像
plt.imshow(cv_rgb)
plt.show()




# 如果img_path路径表示的图像检测到了脸，返回"True" 
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0



human_files_short = human_files[:100]
dog_files_short = train_files[:100]



## TODO: 基于human_files_short和dog_files_short
## 中的图像测试face_detector的表现
human_correct_num = 0.0
dog_correct_num = 0.0


human_correct_num = np.mean([ face_detector(human) for human in human_files_short] )
dog_correct_num = np.mean([ face_detector(dog) for dog in dog_files_short] ) 
    

print( "Accuracy on people images:" "%.2f%%" %(human_correct_num*100)  )
print( "Accuracy on dog images:" "%.2f%%" %(dog_correct_num*100) ) 



from keras.applications.resnet50 import ResNet50

# 定义ResNet50模型
ResNet50_model = ResNet50(weights='imagenet')



from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # 用PIL加载RGB图像为PIL.Image.Image类型
    img = image.load_img(img_path, target_size=(224, 224))
    # 将PIL.Image.Image类型转化为格式为(224, 224, 3)的3维张量
    x = image.img_to_array(img)
    # 将3维张量转化为格式为(1, 224, 224, 3)的4维张量并返回
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)



from keras.applications.resnet50 import preprocess_input, decode_predictions
def ResNet50_predict_labels(img_path):
    # 返回img_path路径的图像的预测向量
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))




def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 




human_files_short = human_files[:100]
dog_files_short = train_files[:100]
### TODO: 测试dog_detector函数在human_files_short和dog_files_short的表现
human_correct_num = 0.0
dog_correct_num = 0.0

human_correct_num = np.mean([dog_detector(human) for human in human_files_short])
dog_correct_num = np.mean([dog_detector(dog) for dog in dog_files_short])
    

print( "Accuracy on people images:" "%.2f%%" %(human_correct_num*100)  )
print( "Accuracy on dog images:" "%.2f%%" %(dog_correct_num*100) ) 




import keras
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# Keras中的数据预处理过程
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255




from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()

### TODO: 定义你的网络架构
model.add( Conv2D( filters = 16, kernel_size = 2, padding = 'same', activation = 'relu', input_shape = (224, 224, 3 )))
model.add( MaxPooling2D( pool_size = 2 ))
model.add( Conv2D( filters = 32, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add( MaxPooling2D( pool_size = 2))
model.add( Conv2D( filters = 64, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add( MaxPooling2D( pool_size = 2 ))
model.add( Flatten() )
model.add( Dense( 500, activation = 'relu' ))
model.add( Dense( 133, activation = 'softmax'))
model.summary()



## 编译模型
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])





from keras.callbacks import ModelCheckpoint  

### TODO: 设置训练模型的epochs的数量

epochs = 10



checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)




## 加载具有最好验证loss的模型

model.load_weights('saved_models/weights.best.from_scratch.hdf5')


# ### 测试模型



# 获取测试数据集中每一个图像所预测的狗品种的index
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# 报告测试准确率
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)



bottleneck_features = np.load('/data/bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']



VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()



VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5', 
                               verbose=1, save_best_only=True)

VGG16_model.fit(train_VGG16, train_targets, 
          validation_data=(valid_VGG16, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)



## 加载具有最好验证loss的模型

VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')


# ### 测试模型

# 获取测试数据集中每一个图像所预测的狗品种的index
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# 报告测试准确率
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


# ### 使用模型预测狗的品种

# In[23]:


from extract_bottleneck_features import *

def VGG16_predict_breed(img_path):
    # 提取bottleneck特征
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # 获取预测向量
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # 返回此模型预测的狗的品种
    return dog_names[np.argmax(predicted_vector)]



### TODO: 从另一个预训练的CNN获取bottleneck特征
bottleneck_features = np.load('/data/bottleneck_features/DogResnet50Data.npz')
train_Resnet50 = bottleneck_features['train']
valid_Resnet50 = bottleneck_features['valid']
test_Resnet50 = bottleneck_features['test']




# In[25]:


np.shape(train_Resnet50)



Resnet50_model = Sequential()
Resnet50_model.add(Flatten(input_shape=train_Resnet50.shape[1:]))
Resnet50_model.add(Dense(20))
#Resnet50_model.add( Dense( 100, activation = 'relu', input_shape = train_Resnet50.shape[1:] ))
Resnet50_model.add( Dense( 133, activation = 'softmax'))

Resnet50_model.summary()



### 编译模型
Resnet50_model.compile( loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])




from keras.callbacks import ModelCheckpoint  



checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Resnet50.hdf5', 
                               verbose=1, save_best_only=True)


Resnet50_model.fit(train_Resnet50, train_targets, 
          validation_data=(valid_Resnet50, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)
Resnet50_model.load_weights('saved_models/weights.best.Resnet50.hdf5')




### 在测试集上计算分类准确率
Resnet50_predictions = [np.argmax(Resnet50_model.predict( np.expand_dims(feature, axis = 0))) for feature in test_Resnet50]

test_accuracy = 100*np.sum(np.array(Resnet50_predictions) == np.argmax( test_targets, axis = 1)) / len(Resnet50_predictions)

print('Test accuracy: %.4f%%' % test_accuracy)


from extract_bottleneck_features import *

def Resnet50_predict_breed( img_path ):
    bottleneck_feature = extract_Resnet50( path_to_tensor(img_path) )
    predict_vector =  Resnet50_model.predict( bottleneck_feature )
    return dog_names[np.argmax(predict_vector)]



def final_function(img_path):
    img = cv2.imread(img_path)
    cv_rgb = cv2.cvtColor( img, cv2.COLOR_BGR2RGB )
    if_human = face_detector( img_path )
    if_dog = dog_detector( img_path )
    if if_dog == 1:
        plt.imshow( cv_rgb )
        plt.show()
        print("Hello, dog!")
        print("You look like a: ")
        print(Resnet50_predict_breed(img_path))
    elif if_human == 1:
        plt.imshow( cv_rgb )
        plt.show()
        print("Hello, human!")
        print("You look like a: ")
        print(Resnet50_predict_breed(img_path))
    else:
        print("Input Error")
        

final_function('Huskie.jpg')
final_function('Labrador.jpg')
final_function('bulldog.jpg')
final_function('sunhonglei.jpg')
final_function('yaoming.jpg')
final_function('huangbo.jpg')
final_function('wuzun.jpg')

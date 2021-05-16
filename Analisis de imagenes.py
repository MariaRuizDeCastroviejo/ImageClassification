#!/usr/bin/env python
# coding: utf-8

# # CLASIFICACIÓN DE IMÁGENES

# Para descargar los datos pulse [aquí.](https://www.kaggle.com/slothkong/10-monkey-species)__

# **Importamos las librerías:**

# In[73]:


# pip install imgaug
# pip install mlxtend  

import os
import cv2
import glob
import h5py
import shutil
import imgaug as aug
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from pathlib import Path
import skimage
from skimage.io import imread
from skimage.transform import resize
from keras.models import Sequential, Model, load_model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from keras import backend as K
import tensorflow as tf


color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', 'InlineBackend.figure_format="svg"')


# **Fijamos la semilla y cargamos los datos:**

# In[6]:


os.environ['PYTHONHASHSEED'] = '0'
seed=123
np.random.seed(seed)

tf.random.set_seed(seed)            # Establecer la semilla aleatoria en tensorflow a nivel de gráfico
aug.seed(seed)                      # Hacer determinista la secuencia de aumento

labels_path = Path('./monkey_labels.txt')


# **Guardamos en un dataframe los distintos tipos de monos que vamos a intentar clasificar:**

# In[7]:


labels_info = []

# leemos el fichero
lines = labels_path.read_text().strip().splitlines()[1:]
for line in lines:
    line = line.split(',')
    line = [x.strip(' \n\t\r') for x in line]
    line[3], line[4] = int(line[3]), int(line[4])
    line = tuple(line)
    labels_info.append(line)
    
# Los pasamos a dataFrame
labels_info = pd.DataFrame(labels_info, columns=['Label', 'Latin Name', 'Common Name', 'Train Images', 'Validation Images'], index=None)
labels_info.head(10)


# In[11]:


print('Entrenamos con:',labels_info['Train Images'].sum(),' imagenes')


# In[12]:


print('Validamos con ', labels_info['Validation Images'].sum(),'imagenes')


# Observamos en la siguiente gráfica que vamos a tratar de clasificar 10 tipos de monos distintos y los set de entrenamiento y validación que vamos a utilizar:

# In[53]:


labels_info.plot.bar(x='Common Name', y=['Train Images', 'Validation Images'])


# Observamos que nuestro set de entrenamiento y de validación no son muy grandes, problablemente habrá que introducir alguna técnica de para evitar el overfitting como DATA AUGMENTATION o TRANSFER LEARNING

# **ANÁLISIS DE IMAGENES PRELIMINAR**

# Mostramos imagenes de los distintos monos que queremos analizar:

# In[76]:


from glob import glob
print("white_headed_capuchin:")#Indicamos que tipo de mono vamos a ver imagenes
multipleImages = glob('./training/n5/**')#aqui introducimos el tipo de mono del que queremos ver imagenes
i_ = 0
plt.rcParams['figure.figsize'] = (10.0, 10.0)
plt.subplots_adjust(wspace=0, hspace=0)
for l in multipleImages[:5]:
    im = cv2.imread(l)
    im = cv2.resize(im, (128, 128)) 
    plt.subplot(5, 5, i_+1) #.set_title(l)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
    i_ += 1


# ## MODELO MACHINE LEARNING: CNN

# In[55]:


train_dir = Path('./training/')
test_dir = Path('./validation/')


# In[57]:


from keras.preprocessing.image import ImageDataGenerator


# Indicamos los parámetros de entrada necesarios para generar nuestro set de imagenes de entrenamiento y validación (imagenes de 150x150)
# 
# Seleccionamos un tamaño de batch de 64,es decir, trabajaremos con 64 muestras antes de actualizar los parámetros internos del modelo.

# In[58]:


height = 150
width = 150
batch_size = 64
seed = 100


# Generamos nuestro training data con 1098 imagenes de entrenamiento:

# In[59]:


# Training generator
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(height, width),
    batch_size=batch_size,
    seed=seed,
    shuffle=True,
    class_mode='categorical')


# Generamos nuestro validation set con 272 imagenes:

# In[60]:


# Test generator
test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(height, width),
    batch_size=batch_size,
    seed=seed,
    shuffle=False,
    class_mode='categorical')


# In[61]:


train_num = train_generator.samples
validation_num = validation_generator.samples 


# Definimos la red convulucional:
# 
# HIDDEN LAYERS
# 
# 1. Primera hiden layer: tomará las imagenes(150x150 y 3 canales, es decir, a color) como input. El mumero de filtros aplicados en esta primera convolución es 32 y de tamaño 3x3(kernel size), es decir, un tensor de 32x3x3. Utilizamos un stride=2 para analizar las imagenes un poco más rapido. Finalmente se usará la función Relu tras la aplicación del tensor.
#     + Tras ello aplicaremos una capa de Batch Normalization: estas capas buscan un preprocesado de la imagen. Las redes neuronales trabajan más rápido si los datos de entrada se mantienen similares entre capa y capa. Para ello se utiliza el BatchNormalization, que ayudará a que nuestra CNN trabaje a mayor velocidad.
#     
# 
# 2. Segunda hidden layer + Batch Normalization: 32x3x3 con función Relu.
# 
# 3. Tercera hidden layer+ Batch Normalization: 64x3x3 con función Relu.
# 
# 4. Cuarta hidden layer+ Batch Normalization: 64x3x3 con función Relu.
# 
# 5. Flatten layer: 512x1x1. el contenido de los feature maps obtenidos gracias a las anteriores capas se convierte en un Tensor unidimensional que se puede utilizar en las capas Dense. Añadimos la duncion de activación Relu tras aplicar el tensor unidimensional
# 
# CLASSIFICATION: DENSE LAYER
# 
# 6. Aplicaremos un último tensor unidimensional de 10x1x1 + Un average pooling que reducirá el tamaño del vector a clasificar + una función de activación Softmax que clasificara finalmente la imagen.

# In[63]:


def get_net(num_classes):
    from keras.models import Sequential
    from keras.layers import Conv2D, Activation, BatchNormalization, GlobalAvgPool2D, MaxPooling2D, Dropout

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), strides=2))
    model.add(Activation('relu'))

    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), strides=2))
    model.add(Activation('relu'))

    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), strides=2))
    model.add(Activation('relu'))

    model.add(Conv2D(512, (1, 1), strides=2))
    model.add(Activation('relu'))
    model.add(Conv2D(num_classes, (1, 1)))
    model.add(GlobalAvgPool2D())
    model.add(Activation('softmax'))
    return model

num_classes = 10
net = get_net(num_classes)
net.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])
net.summary()


# Como podemos observar nuestro output de 8x8x10 se convierte en un vector unidimensional de 10 antes de entrar en la última funcion de activación.

# Ahora entrenamos nuestra red convolucional con 10 epochs:

# In[66]:


from keras.callbacks import ModelCheckpoint, EarlyStopping
filepath=("monkey.h5f")
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# = EarlyStopping(monitor='val_acc', patience=15)
callbacks_list = [checkpoint]#, stopper]

epochs = 10

history = net.fit_generator(train_generator,
                              steps_per_epoch= train_num // batch_size,
                              epochs=epochs,
                              validation_data=train_generator,
                              validation_steps= validation_num // batch_size,
                              callbacks=callbacks_list, 
                              verbose = 1
                             )


# Podemos apreciar que el accuracy alcanzado con 10 epochs no es muy bueno, probablemente cambiando la estructura de la red convulucional añadiendo más capas o realizando más epochs mejoraría el resultado a decrimento de tiempo de computación. 
# 
# Que el accuracy con 10 epochs sea del 0.508 quiere decir que acertaremos en la clasificación de los monos un 50% de las veces, lo que no es un valor muy fiable. 
# 
# Para mejorar este accuracy proponemos realizar un modelo de transfer learning con data augmantation. Ya que nuestro modelo no entrena con suficientes imagenes y de ahí que no obtenga resultados muy satisfactorios.

# In[67]:


def visualized_history(history):
    
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    f,ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].plot(epochs, loss)
    ax[0].plot(epochs, val_loss)
    ax[0].set_title("Función de pérdida")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("loss")
    ax[0].legend(['train', 'validation'])

    ax[1].plot(epochs, acc)
    ax[1].plot(epochs, val_acc)
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("accuracy")
    ax[1].legend(['train', 'validation'])
    
    plt.show()
    
visualized_history(history)


# En estas gráficas confirmamos lo explicado anteriormente. Si exportaramos el modelo realizado con las imagenes de test el accuracy sería muy malo.
# 
# No es un modelo que pudieramos poner en producción con solo 10 epochs o con tan pocas imagenes de entrenamiento.

# ## MODELO APLICANDO DATA AUGMENTATION Y TRANSFER LEARNING

# Creamos un diccionario con el número de clases; como son 10 llegamos hasta 9 y creamos otro también para cada nombre de cada clase:

# In[49]:


training_data = Path('./training/') 
validation_data = Path('./validation/') 


# In[13]:


labels_dict= {'n0':0, 'n1':1, 'n2':2, 'n3':3, 'n4':4, 'n5':5, 'n6':6, 'n7':7, 'n8':8, 'n9':9}

names_dict = dict(zip(labels_dict.values(), labels_info["Common Name"]))
print(names_dict)


# - **Generamos la muestra de Train:**

# In[14]:


# Entrenamiento
train_df = []
for folder in os.listdir(training_data):
    
    imgs_path = training_data / folder          # Carpeta de imágenes
    imgs = sorted(imgs_path.glob('*.jpg'))      # Imágenes del directorio
    
    for img_name in imgs:
        train_df.append((str(img_name), labels_dict[folder]))


train_df = pd.DataFrame(train_df, columns=['image', 'label'], index=None)
train_df = train_df.sample(frac=1.).reset_index(drop=True)                     # Generamos muestra aleatoria


# - **Generamos la muestra de Test:**

# In[15]:


# Validación
valid_df = []
for folder in os.listdir(validation_data):
    imgs_path = validation_data / folder
    imgs = sorted(imgs_path.glob('*.jpg'))
    for img_name in imgs:
        valid_df.append((str(img_name), labels_dict[folder]))

        
valid_df = pd.DataFrame(valid_df, columns=['image', 'label'], index=None)
valid_df = valid_df.sample(frac=1.).reset_index(drop=True)                 # Generamos muestra aleatoria


# In[16]:


print("Número de muestras en el fichero de Train: ", len(train_df))
print("Número de muestras en el fichero de Test: ", len(valid_df))

print("\n",train_df.head(), "\n")
print("=================================================================\n")
print("\n", valid_df.head())


# - **Constantes que se van a utilizar:**

# In[17]:


# Dimensiones de las imágenes y 3 canales que indica que son imagenes a color
img_rows, img_cols, img_channels = 224,224,3

# Batch size de la muestra de entrenamiento
batch_size=8

# Número de clases del dataset
nb_classes=10


# # Data Augmentation

# Cuando entrenamos redes convulucionales con muy pocas imagenes de entrenamiento como es nuestro caso es muy fácil caer en overfitting, en nuestro caso observavamos que con 10 epochs el modelo de entrenamiento tiene un accuracy del 50% que presumiblemente si aumentamos las epochs mejoraría el accuaracy. En cambio, en las gráficas observavamos que se producía overfitting, ya que no era un modelo exportable en el caso de test donde el modelo actuaba muy mal.
# 
# La técnica de Data augmentation evitará que caigamos en overfitting aumentando las imagenes de muestra con transformaciones aleatorias.De esta manera, nuestro modelo nunca verá más de dos imagenes iguales, esto permitirá generalizar mejor al modelo.
# 
# Cuantos más datos proporcionemos, mejor será el rendimiento (hasta llegar a un límite). Por eso es imporante el aumento de datos. Usaremos imgaug para aumentar nuestras imágenes:

# In[18]:


# Secuencia de aumentos para cada imagen
seq = iaa.OneOf([
    iaa.Fliplr(),                 # Volteos
    iaa.Affine(rotate=20),        # Rotación
    iaa.Multiply((1.2, 1.5))])    # Brillo aleatorio


# # Data Generator

# In[19]:


def data_generator(data, batch_size, is_validation_data=False):
    n = len(data)                                     # Número de observaciones
    nb_batches = int(np.ceil(n/batch_size))

    indices = np.arange(n)
    
    # Concatenamos los datos y las etiquetas
    batch_data = np.zeros((batch_size, img_rows, img_cols, img_channels), dtype=np.float32)
    batch_labels = np.zeros((batch_size, nb_classes), dtype=np.float32)
    
    while True:
        if not is_validation_data:
            np.random.shuffle(indices)
            
        for i in range(nb_batches):
            next_batch_indices = indices[i*batch_size:(i+1)*batch_size]
            
            for j, idx in enumerate(next_batch_indices):
                img = cv2.imread(data.iloc[idx]["image"])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                label = data.iloc[idx]["label"]
                
                if not is_validation_data:
                    img = seq.augment_image(img)
                
                img = cv2.resize(img, (img_rows, img_cols)).astype(np.float32)
                batch_data[j] = img
                batch_labels[j] = to_categorical(label,num_classes=nb_classes)
            
            batch_data = preprocess_input(batch_data)
            yield batch_data, batch_labels


# In[20]:


# Generador de datos para la muestra de entrenamiento
train_data_gen = data_generator(train_df, batch_size)

# Generador de datos para la muestra de validación
valid_data_gen = data_generator(valid_df, batch_size, is_validation_data=True)


# # Modelling

# Las redes preentrenadas con grandes cantidades de datos mejoran el accuracy de los modelos cuando estos trabajan con pequelos datasets de datos como es nuestro caso.
# 
# 
# Elegimos la red conv VGG16 entreada en ImageNet como red base para hacer el aprendizajepor transferencia. Este tipo de redes convolucionales muy profundas se utilizan mucho para el reconocimiento de imágenes a gran escala:

# In[23]:


# Modelo base
def get_base_model():
    base_model = VGG16(input_shape=(img_rows, img_cols, img_channels), weights='imagenet', include_top=True)#Incluimos un clasificador densamente conectado
    return base_model


# Definimos la red convulucional:
# 
# HIDDEN LAYERS
# 
# Primeras hidden layer: estan compuesta por la misma estructura de dos tensores de 224x224x64 después se aplica una capa de pooling que reduce a la mitad las dimensiones del input en la siguiente capa. La funcion de activación que se utiliza en estas capas es Softmax.
# 
# Siguientes hidden layer: estan compuesta por la misma estructura de tres tensores que empiezan con una dimensión de 56x56x128 después se aplica una capa de pooling que reduce a la mitad las dimensiones del input en la siguiente capa. La funcion de activación que se utiliza en estas capas es Softmax.
# 
# 
# Flatten layer: 512x1x1. el contenido de los feature maps obtenidos gracias a las anteriores capas se convierte en un Tensor unidimensional que se puede utilizar en las capas Dense. Añadimos la duncion de activación Relu tras aplicar el tensor unidimensional
# 
# CLASSIFICATION: DENSE LAYER
# 
# Con la técnica de flatten pasamos del poolin de 5x5x512 a un vector unidimensional de 25088.
# 
# Aplicamos tres capas densas donde todas las neuronas estas conectadas entre sí hasta llegar a un vector unidimensional de 10 que clasificará las imagenes.
# 
# En estas capas densas observamos que se utiliza la técnica de DropOut con una probabilidad de 0.7, esta técnica restringe la adaptación de la red a los datos en el momento del entrenamiento para evitar el overfitting. La probabilidad de 0.7 representa el porcentaje de neuronas cuya salida se reducirá
# al pasar a la siguiente capa. Esto reduce efectivamente el número de parámetros y simplifica el modelo que se está ajustando.

# In[24]:


base_model = get_base_model()

# Obtenemos la salida de la penúltima capa densa
base_model_output = base_model.layers[-2].output

# Añadimos nuevas capas
x = Dropout(0.7,name='drop2')(base_model_output) #Destacar que utilizamos la técnica de dropout para evitar overfitting
output = Dense(10, activation='softmax', name='fc3')(x)

# Definimos un nuevo modelo
model = Model(base_model.input, output)

# Congelamos todas las capas del modelo base
for layer in base_model.layers[:-1]:
    layer.trainable=False

# Modelo final
optimizer = RMSprop(0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()


# In[27]:


# Carga los pesos de la mejor iteración una vez que finaliza el entrenamiento
es = EarlyStopping(patience=10, restore_best_weights=True)

# Checkpoint pra guardar el modelo
chkpt = ModelCheckpoint(filepath="model1", save_best_only=True)

nb_train_steps = int(np.ceil(len(train_df)/batch_size))
nb_valid_steps = int(np.ceil(len(valid_df)/batch_size))

# Número de epochs: con 3 ya conseguimos una precisiónmuy buena
nb_epochs=3


# Eligimos 3 epochs y observamos que con un modelo de transfer Learning entrenado con la red VG16 en la primera epoch ya obtenemos un accuracy en el training cercano a uno.
# 
# Se aprecia de forma drástica la mejora de un modelo a otro.

# In[29]:


# Entrenamos el modelo
history1 = model.fit_generator(train_data_gen, 
                              epochs=nb_epochs, 
                              steps_per_epoch=nb_train_steps, 
                              validation_data=valid_data_gen, 
                              validation_steps=nb_valid_steps,
                              callbacks=[es,chkpt])


# - **Función de pérdida y precisión del modelo en función de los epoch:**

# In[30]:


# Precisión
train_acc = history1.history['accuracy']
valid_acc = history1.history['val_accuracy']

# Función de pérdida
train_loss = history1.history['loss']
valid_loss = history1.history['val_loss']

# Número de entradas
xvalues = np.arange(len(train_acc))


# In[31]:


# Gráfico
f,ax = plt.subplots(1,2, figsize=(10,5))
ax[0].plot(xvalues, train_loss)
ax[0].plot(xvalues, valid_loss)
ax[0].set_title("Función de pérdida")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("loss")
ax[0].legend(['train', 'validation'])

ax[1].plot(xvalues, train_acc)
ax[1].plot(xvalues, valid_acc)
ax[1].set_title("Accuracy")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("accuracy")
ax[1].legend(['train', 'validation'])

plt.show()


# Observamos que con transfer Learning + Data Augmentation + DropOut tambiénm evitamos el overfitting y obtenemos un accuracy muy aceptable con tan solo 3 epochs.

# Pérdida y ajuste final:

# In[32]:


valid_loss, valid_acc = model.evaluate_generator(valid_data_gen, steps=nb_valid_steps)
print(f"Accuracy del modelo final: {valid_acc*100:.2f}%")


# # Model Interpretability

# Visualización de los resultados de las capas intermedias. El objetivo es mostrar todos los mapas de activación de cada bloque de convolución para cualquier imagen de muestra de validación:

# In[28]:


# Capas para las que se desea visualizar las salidas
outputs = [layer.output for layer in model.layers[1:18]]

# Define a new model that generates the above output
vis_model = Model(model.input, outputs)

# Comprobamos si tenemos todas las capas que necesitamos para la visualización
vis_model.summary()


# Guardamos los nombres de todas estas capas intermedias en una lista:

# In[29]:


# Guardamos los nombres de las capas que nos interesan
layer_names = []
for layer in outputs:
    layer_names.append(layer.name.split("/")[0])

    
print("Capas que se van a visualizar: ")
print(layer_names)


# Mapa de calor para una predicción de una imagen. El arguemnto que se le pasa es cualquier imagen que haya sido preprocesada con el método preprocess_input () con la etiqueta que ha sido predicha por la red para esta imagen. Devuelve un mapa de calor generado sobre la última salida de la capa de convolución:

# In[30]:


def get_CAM(processed_image, predicted_label):

    # Activaciones para la etiqueta predicha
    predicted_output = model.output[:, predicted_label]
    
    # Última capa de conv en tu modelo
    last_conv_layer = model.get_layer('block5_conv3')
    
    # Gradients wrt de la última capa de conv
    grads = K.gradients(predicted_output, last_conv_layer.output)[0]        #Error
    
    # Gradiente medio por mapa de características
    grads = K.mean(grads, axis=(0,1,2))
    
    # Función que genere los valores para la salida y los gradientes
    evaluation_function = K.function([model.input], [grads, last_conv_layer.output[0]])
    
    # Obtener los valores
    grads_values, conv_ouput_values = evaluation_function([processed_image])
    
    # Iterar sobre cada mapa de características y multiplicar los valores de gradiente con los valores de salida de conv
    for i in range(512):                                  # 512 funciones en nuestra ultima capa de conv
        conv_ouput_values[:,:,i] *= grads_values[i]
    
    # Mapa de calor
    heatmap = np.mean(conv_ouput_values, axis=-1)
    
    # Quitamos los valores negativos
    heatmap = np.maximum(heatmap, 0)
    
    # Normalizamos
    heatmap /= heatmap.max()
    
    return heatmap


# Seleccionamos una muestra aleatoria del marco de datos de validación y generamos una predicción para el mismo. También almacena el mapa de calor y las capas de activación intermedias. 
# 
#         idx: índice aleatorio para seleccionar una muestra de los datos de validación
#   
#         activaciones: valores de activación para capas intermedias

# In[43]:


def show_random_sample(idx):


    # Seleccionamos la muestra y leemos la imagen y la etiqueta correspondientes
    sample_image = cv2.imread(valid_df.iloc[idx]['image'])
    sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
    sample_image = cv2.resize(sample_image, (img_rows, img_cols))
    sample_label = valid_df.iloc[idx]["label"]
    
    # Pre procesamos la imagen
    sample_image_processed = np.expand_dims(sample_image, axis=0)
    sample_image_processed = preprocess_input(sample_image_processed)
    
    # Generamos las capas activación intermedias utilizando el modelo de visualización
    activations = vis_model.predict(sample_image_processed)
    
    # Obtenemos la etiqueta predicha por nuestro modelo original
    pred_label = np.argmax(model.predict(sample_image_processed), axis=-1)[0]
    
    # Mapa de activación aleatorio
    sample_activation = activations[0][0,:,:,32]
    
    # Normalizamos el mapa de activación
    sample_activation-=sample_activation.mean()
    sample_activation/=sample_activation.std()
    
    # Convertimos los valores de los píxeles a valores entre 0 y 255
    sample_activation *=255
    sample_activation = np.clip(sample_activation, 0, 255).astype(np.uint8)
    
   
    # Mapa de calor para el mapa de activación de clases (CAM)
    heatmap = get_CAM(sample_image_processed, pred_label)
    heatmap = cv2.resize(heatmap, (sample_image.shape[0], sample_image.shape[1]))
    heatmap = heatmap *255
    heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    super_imposed_image = heatmap * 0.5 + sample_image
    super_imposed_image = np.clip(super_imposed_image, 0,255).astype(np.uint8)

    f,ax = plt.subplots(2,2, figsize=(15,8))
    ax[0,0].imshow(sample_image)
    ax[0,0].set_title(f"True label: {sample_label} \n Predicted label: {pred_label}")
    ax[0,0].axis('off')
    
    ax[0,1].imshow(sample_activation)
    ax[0,1].set_title("Random feature map")
    ax[0,1].axis('off')
    
    ax[1,0].imshow(heatmap)
    ax[1,0].set_title("Class Activation Map")
    ax[1,0].axis('off')
    
    ax[1,1].imshow(super_imposed_image)
    ax[1,1].set_title("Activation map superimposed")
    ax[1,1].axis('off')
    plt.show()
    
    return activations


# El error que sale:
# 
# Calling `Model.predict` in graph mode is not supported when the `Model` instance was constructed with eager mode enabled. Please construct your `Model` instance in graph mode or call `Model.predict` with eager mode enabled.

# In[ ]:


# Activaciones intermedias y mapa de calor
activations= show_random_sample(123)


# Consulte la función visualize_intermediate_activations () para obtener más detalles. Para CAM, tomaremos la misma imagen de muestra y obtendremos el resultado de la última capa de convolución. También calculamos los gradientes para esta capa que usaremos para generar un mapa de calor. Consulte la función get_CAM () para obtener más detalles.

#     Esta función se utiliza para visualizar todos los mapas de activación inmediatos.
#     
#     Argumentos:
#         layer_names: lista de nombres de todas las capas intermedias que elegimos
#         activaciones: todos los mapas de activación intermedios

# In[35]:


def visualize_intermediate_activations(layer_names, activations):

    assert len(layer_names)==len(activations), "Make sure layers and activation values match"
    images_per_row=16
    
    for layer_name, layer_activation in zip(layer_names, activations):
        nb_features = layer_activation.shape[-1]
        size= layer_activation.shape[1]

        nb_cols = nb_features // images_per_row
        grid = np.zeros((size*nb_cols, size*images_per_row))

        for col in range(nb_cols):
            for row in range(images_per_row):
                feature_map = layer_activation[0,:,:,col*images_per_row + row]
                feature_map -= feature_map.mean()
                feature_map /= feature_map.std()
                feature_map *=255
                feature_map = np.clip(feature_map, 0, 255).astype(np.uint8)

                grid[col*size:(col+1)*size, row*size:(row+1)*size] = feature_map

        scale = 1./size
        plt.figure(figsize=(scale*grid.shape[1], scale*grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(grid, aspect='auto', cmap='viridis')
    plt.show()


# In[ ]:


# visualize all the activation maps for this sample
visualize_intermediate_activations(activations=activations, layer_names=layer_names)


# In[ ]:


sample_idx=200
activations= show_random_sample(sample_idx)


# In[ ]:


sample_idx=10
activations= show_random_sample(sample_idx)


# In[ ]:


sample_idx=55
activations= show_random_sample(sample_idx)


# In[ ]:


sample_idx=70
activations= show_random_sample(sample_idx)


# In[ ]:





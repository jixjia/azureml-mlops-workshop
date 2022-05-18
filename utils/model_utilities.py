import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
from sklearn.model_selection import train_test_split
from imutils.paths import list_images
from . import video_utilities as vu

# TF/Keras
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D

''' Prepare Pair-wise train/val dataset for training a Siamese Network '''

def prep_dataset(trainPath, imageShape):
    # load ID dataset and scale the pixel values to the range of [0, 1]
    print('[INFO] Loading dataset...', end='')

    imagePaths = list(list_images(trainPath))
    labels = []
    images = []

    for path in imagePaths:
        # resize to network input shape
        image = cv2.imread(path)
        image = cv2.resize(image, (imageShape[0], imageShape[1]), interpolation = cv2.INTER_AREA)
        image = img_to_array(image)
        
        # normalize to unit scale
        image = np.array(image, dtype='float')/255.0
        
        # save np array images and labels
        images.append(image)
        labels.append(os.path.split(os.path.split(path)[0])[-1])  

    print(f'Done ({len(images)} loaded)')

    # test/train split
    trainX, testX, trainY, testY = train_test_split(images, labels, test_size=0.2)

    # prepare the positive and negative pairs
    print('[INFO] Preparing positive and negative pairs...', end='')

    (pairTrain, labelTrain), triplets = make_pairs(trainX, trainY)
    (pairTest, labelTest), triplets = make_pairs(testX, testY)
    print('Done')
    print(f'[INFO] Train/Test paris = {len(pairTrain[:,0])}/{len(pairTest[:,0])} (Input shape: {pairTrain[0, 0].shape}/{pairTest[0, 1].shape})')

    return pairTrain, labelTrain, pairTest, labelTest

 
def make_pairs(images, labels):
    # initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive or negative
    pairImages = []
    pairLabels = []
    triplet = []

    # calculate the total number of classes present in the dataset
    # and then build a list of indexes for each class label that
    # provides the indexes for all examples with a given label
    # numClasses = len(np.unique(labels))
    # idx = [np.where(labels == i)[0] for i in range(0, numClasses)]

    # loop over all images
    for idx in range(len(images)):
        # grab the current image and label belonging to the current
        # iteration
        currentImage = images[idx]
        currentLabel = labels[idx]

        # randomly pick an image that belongs to the *same* class
        # label
        idxPos = [i for i,elem in enumerate(labels) if elem == currentLabel]
        posImage = images[random.choice(idxPos)]

        # prepare a positive pair and update the images and labels
        # lists, respectively
        pairImages.append([currentImage, posImage])
        pairLabels.append([1])

        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        idxNeg = [i for i,elem in enumerate(labels) if elem != currentLabel]
        negImage = images[random.choice(idxNeg)]

        # prepare a negative pair of images and update our lists
        pairImages.append([currentImage, negImage])
        pairLabels.append([0])

        triplet.append((currentImage, posImage, negImage))

    # return a 2-tuple of our image pairs and labels
    return (np.array(pairImages), np.array(pairLabels)), triplet


'''Build Transfer Learning Network Architecture'''

def build_model(imageShape=(224,224,3), initial_lr=1e-4, epochs=50):
    print('[INFO] Extracting image feature vectors using pre-trained siamese network...')
    imgA = Input(shape=imageShape)
    imgB = Input(shape=imageShape)
    featureExtractor = build_siamese_model(imageShape)
    featsA = featureExtractor(imgA)
    featsB = featureExtractor(imgB)

    # network arthictecture
    print(featureExtractor.summary())
    
    # compute euclidean distance from the two image feature vecs in the Keras lambda layer 
    distance = Lambda(euclidean_distance)([featsA, featsB])

    # build the full siamese network using the image feature vec as input and distance as output
    print('[INFO] Compiling model...', end='')
    model = Model(inputs=[imgA, imgB], outputs=distance)
    optimizer = Adam(lr=initial_lr,  decay=initial_lr / epochs)
    model.compile(loss=contrastive_loss, optimizer=optimizer, metrics=['accuracy'])
    print('Done')

    return model
    

def build_siamese_model(inputShape, embeddingDim=64):
	# specify the inputs for the feature extractor network
	inputs = Input(inputShape)

	# define the first set of CONV => RELU => POOL => DROPOUT layers
	x = Conv2D(64, (2, 2), padding='same', activation='relu')(inputs)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Dropout(0.25)(x)
	
	# second set of CONV => RELU => POOL => DROPOUT layers
	x = Conv2D(64, (2, 2), padding='same', activation='relu')(x)
	x = MaxPooling2D(pool_size=2)(x)
	x = Dropout(0.2)(x)

	# third set of CONV => RELU => POOL => DROPOUT layers
	x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
	x = MaxPooling2D(pool_size=2)(x)
	x = Dropout(0.1)(x)

	# prepare the final outputs
	x = GlobalAveragePooling2D()(x)
	outputs = Dense(embeddingDim)(x)

	# build the model
	return Model(inputs, outputs)


def contrastive_loss(y, preds, margin=0.5):
    # explicitly cast the true class label data type to the predicted class label
    y = tf.cast(y, preds.dtype)

    # calculate the contrastive loss between the true labels and the predicted labels
    squaredPreds = K.square(preds)
    squaredMargin = K.square(K.maximum(margin - preds, 0))
    loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)

    # return the computed contrastive loss to the calling function
    return loss

   
def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors

    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1,
        keepdims=True)

    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def plot_training(H, plotPath):
    # construct a plot that plots and saves the training history
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(H.history['loss'], label='train_loss')
    plt.plot(H.history['val_loss'], label='val_loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.legend(loc='lower left')
    plt.savefig(plotPath)


''' Inference '''

def hconcat_resize(im_list):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=cv2.INTER_CUBIC)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


def test_inference(testPath, model, imageShape=(224,224,3)):
    correct = 0
    total = 0
    test_accuracy = 0
    output_images = []
    
    # load templates
    print("[INFO] Loading templates...", end='')
    char_name = os.path.basename(testPath)
    template_path = os.path.join(testPath.replace(char_name,''), f'templates/{char_name}')
    templates = []
    labels = []
    templatePaths = list(list_images(template_path))

    for path in templatePaths:
        file = os.path.basename(path)
        if file.startswith('template_'):
            template_name = os.path.basename(path)
            template = cv2.imread(path)
            template = cv2.resize(template, (imageShape[0], imageShape[1]), interpolation = cv2.INTER_AREA)
            template = img_to_array(template)
            template = np.array(template, dtype='float')/255.0
            template = np.expand_dims(template, axis=0)

            # save normalized ID template and labels
            templates.append((template, template_name))
            labels.append(template_name.replace('template_','').split('.')[0])

    print(f'Done ({len(templates)} loaded)')

    # load query images
    print("[INFO] Recognizing query images...")

    for path in os.listdir(testPath):
        # ignore non-image files
        if os.path.basename(path).split('.')[-1] in ['jpg','png','gif','bmp','tiff','webp']:
            total += 1
            query_label = os.path.basename(path).split('_')[0]
            image = cv2.imread(os.path.join(testPath, path))
            resize = cv2.resize(image, (imageShape[0], imageShape[1]), interpolation=cv2.INTER_AREA)
            resize = img_to_array(resize)
            resize = np.array(resize, dtype='float')/255.0
            resize = np.expand_dims(resize, axis=0)

            # perform One-Shot Learning on query images by feeding it through the trained SN
            results = []

            for idx, (template_object, label) in enumerate(zip(templates, labels)):
                (template, template_name) = template_object
                preds = model.predict([template, resize])
                proba = preds[0][0]
                
                results.append((label, proba, template_name))

            # sort difference score in asc order    
            results.sort(key=lambda x: x[1], reverse=False)

            # get best match result
            best_match = results[0][0]
            best_matched_template = results[0][2]

            if best_match == query_label:
                correct += 1 
            
            # horizontally stack query and template images for saving
            tempalte_img = cv2.imread(os.path.join(template_path, best_matched_template))
            output_images.append(hconcat_resize([image, tempalte_img]))
    
    test_accuracy = round(correct / total, 3) if total > 0 else 0

    return output_images, test_accuracy
            


def run_inference(testPath, model, imageShape=(224,224,3)):
    # load templates
    print("[INFO] Loading templates...", end='')
    char_name = os.path.basename(testPath)
    template_path = os.path.join(testPath.replace(char_name,''), f'templates/{char_name}')
    templates = []
    labels = []
    templatePaths = list(list_images(template_path))

    for path in templatePaths:
        file = os.path.basename(path)
        if file.startswith('template_'):
            template_name = os.path.basename(path)
            template = cv2.imread(path)
            template = cv2.resize(template, (imageShape[0], imageShape[1]), interpolation = cv2.INTER_AREA)
            template = img_to_array(template)
            template = np.array(template, dtype='float')/255.0
            template = np.expand_dims(template, axis=0)

            # save normalized ID template and labels
            templates.append((template, template_name))
            labels.append(template_name.replace('template_','').split('.')[0])

    print(f'Done ({len(templates)} loaded)')

    # load query images
    print("[INFO] Recognizing query images...")

    for path in os.listdir(testPath):
        # ignore non-image files
        if os.path.basename(path).split('.')[-1] in ['jpg','png','gif','bmp','tiff','webp']:
            image = cv2.imread(os.path.join(testPath, path))
            resize = cv2.resize(image, (imageShape[0], imageShape[1]), interpolation=cv2.INTER_AREA)
            resize = img_to_array(resize)
            resize = np.array(resize, dtype='float')/255.0
            resize = np.expand_dims(resize, axis=0)

            # perform One-Shot Learning on query images by feeding it through the trained SN
            results = []

            for idx, (template_object, label) in enumerate(zip(templates, labels)):
                (template, template_name) = template_object
                preds = model.predict([template, resize])
                proba = preds[0][0]
                
                results.append((label, proba, template_name))

            # sort difference score in asc order    
            results.sort(key=lambda x: x[1], reverse=False)

            # get best match result
            best_match = results[0][0]
            best_matched_template = results[0][2]
            
            # visualize result
            tempalte_img = cv2.imread(os.path.join(template_path, best_matched_template))
            images = [image, tempalte_img]
            vu.show_images(images, ['Query', 'Template'], width=10, height=10)

            print('Best Match:', best_match)
            print('Candidates:', results, '\n')

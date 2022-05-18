# import the necessary packages
from PIL import ImageFont, ImageDraw, Image
from matplotlib import pyplot as plt
from urllib.request import urlopen
import cv2
import datetime
import math
import numpy as np


def crop_square_center(image, cropX, cropY):
    h, w = image.shape[:2]
    startx = w//2-(cropX//2)
    starty = h//2-(cropY//2)
    return image[starty:starty+cropY, startx:startx+cropX]


def trim_and_save(image, imageName, bbox, jpegQuality=80):
    # timestamp (up to millisecond)
    timenow = datetime.datetime.now().strftime('%H%M%S.%f')[:-2]
    fileName = f'{imageName}-{timenow}.jpg'

    # trim ROI
    outputImage = trim(image, bbox) if bbox else image
    
    # save as JPEG quality
    outputPath = f'output/{fileName}'
    cv2.imwrite(outputPath, outputImage, [int(cv2.IMWRITE_JPEG_QUALITY), jpegQuality])
    return outputImage, outputPath
    

def trim(image, bbox):
    if bbox:
        # crop image using bbox and margin
        (startX, startY, endX, endY) = safe_bbox(image, bbox)
        return image[startY:endY, startX:endX]
    else:
        return image


def get_portrait_bbox(face_bbox, face_to_portrait_ratio, portarit_aspect):
    (startX, startY, endX, endY) = face_bbox
    faceH = endY-startY
    faceW = endX-startX
    centroidX = int(startX+faceW/2)
    centroidY = int(startY+faceH/2)*.99
    
    portraitW = np.sqrt(faceH*faceW / (portarit_aspect*face_to_portrait_ratio))
    portraitH = portraitW * portarit_aspect
    portraitStartX = int(centroidX - portraitW/2)
    portraitStartY = int(centroidY - portraitH/2)
    portraitEndX = int(centroidX + portraitW/2)
    portraitEndY = int(centroidY + portraitH/2)

    return (portraitStartX, portraitStartY, portraitEndX, portraitEndY)


def expand_bbox(image, bbox, expand_ratio=0.1):
    h, w = image.shape[:2]
    offsetX = w * expand_ratio
    offsetY = h * expand_ratio
    (startX, startY, endX, endY) = bbox
    startX = startX - offsetX
    startY = startY - offsetY
    endX = endX + offsetX
    endY = endY + offsetY
    return safe_bbox(image, (startX, startY, endX, endY))


def safe_bbox(image, bbox):
    # set bbox coords to range from 0 ~ max height/width of the image
    h, w = image.shape[:2]
    startX = int(min(max(bbox[0], 0), w))
    startY = int(min(max(bbox[1], 0), h))
    endX = int(min(max(bbox[2], 0), w))
    endY = int(min(max(bbox[3], 0), h))
    return (startX, startY, endX, endY)


def draw_bbox_with_label(image, label, bbox, color=(0,200,0)):
    (startX, startY, endX, endY) = safe_bbox(image, bbox)
    
    # draw rectangle using bbox
    cv2.rectangle(image, (startX, startY), (endX, endY), color, 3) 

    # text label inside top-left of bbox
    if label and len(label) > 1:
        labelHeight = 12
        
        # draw multi-line text label
        for enum, textline in enumerate(label.split('\n')):
            labelWidth = len(textline)*7

            # determine whether to show text label above or inside the bbox
            if startY < labelHeight:
                labelPositionY = startY+labelHeight*enum
                textPositionY = startY+labelHeight*(enum+1)-2
            else:
                labelPositionY = startY-labelHeight*(enum+1)
                textPositionY = startY-labelHeight*enum-2

            cv2.rectangle(image, (startX-2, labelPositionY), (startX+labelWidth, labelPositionY+labelHeight), color, -1)
            cv2.putText(image, textline, (startX, textPositionY), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

    return image


# Display multiple images in Matplotlib subplots
def show_images(images, titles=None, height=16, width=16, axis='on'):
    
    # define plot matrix layout
    n = len(images)
    if n == 0:
        return None
    elif n >= 4:
        nCols = 4
        nRows = math.ceil(n/4)
    else:
        nCols = n
        nRows = math.ceil(n/3)
        
    # create figure object
    fig = plt.figure(figsize=(height, width))
    
    # for each image
    for i, image in enumerate(images):
        # convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # add subplots to figure
        fig.add_subplot(nRows, nCols, i+1)
        plt.axis(axis)
        
        # add subtitles
        if titles is not None:
            plt.gca().set_title(titles[i]) 

        # add images to two subplots
        plt.imshow(image_rgb)
    
    # show entire plot
    plt.show()
    return None
    
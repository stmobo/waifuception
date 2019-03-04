import pandas as pd
import math
import numpy as np
import tensorflow as tf
import sys
from PIL import Image

from tensorflow.keras.applications import inception_v3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras import callbacks
from tensorflow.keras import metrics
from tensorflow.keras import backend as K

gender_tags = ['female', 'male']
rating_classes = ['safe', 'questionable', 'explicit']

hair_color_tags = ['aqua_hair', 'black_hair', 'blonde_hair', 'blue_hair', 'brown_hair', 'green_hair', 'grey_hair', 'orange_hair', 'pink_hair', 'purple_hair', 'red_hair', 'silver_hair', 'white_hair', 'multicolored_hair']
hair_length_tags = ['very_short_hair', 'short_hair', 'medium_hair', 'long_hair', 'very_long_hair', 'bald']
hair_style_tags = ['curly_hair', 'drill_hair', 'flipped_hair', 'hair_flaps', 'messy_hair', 'pointy_hair', 'ringlets', 'spiked_hair', 'wavy_hair', 'bangs', 'ahoge', 'braid', 'hair_bun', 'ponytail', 'twintails']

eye_color_tags = ['aqua_eyes', 'black_eyes', 'blue_eyes', 'brown_eyes', 'green_eyes', 'grey_eyes', 'orange_eyes', 'pink_eyes', 'purple_eyes', 'red_eyes', 'silver_eyes', 'white_eyes', 'yellow_eyes', 'heterochromia']
eye_misc_tags = ['closed_eyes', 'one_eye_closed', 'glasses']

expression_tags = ['angry', 'annoyed', 'blush', 'embarrassed', 'bored', 'confused', 'crazy', 'expressionless', 'happy', 'nervous', 'pout', 'sad', 'scared', 'worried', 'serious', 'sigh', 'sleepy', 'thinking', 'ahegao']

breast_tags = ['flat_chest', 'small_breasts', 'medium_breasts', 'large_breasts', 'huge_breasts', 'gigantic_breasts']
ass_tags = ['ass', 'flat_ass', 'huge_ass']
pose_tags = ['kneeling', 'lying', 'sitting', 'standing']

attire_tags = [
    'hat',
    'hairband',
    'hair_bow',
    'hair_ribbon',
    'helmet',
    'shirt',
    'dress',
    'jacket',
    'vest',
    'pants',
    'shorts',
    'skirt',
    'jeans',
    'pantyhose',
    'socks',
    'thighhighs',
    'shoes',
    'boots',
    'sandals',
    'uniform',
    'apron',
    'armor',
    'cape',
    'hood',
    'school_uniform',
    'serafuku',
    'suit',
    'swimsuit',
    'bikini',
    'japanese_clothes',
    'kimono',
    'earrings',
    'hair_ornament',
    'hairclip',
    'detached_sleeves',
    'gloves',
    'fingerless_gloves',
    'elbow_gloves',
    'belt',
    'headphones_around_neck',
    'goggles_around_neck',
    'necklace',
    'necktie',
    'scarf'
]

# list of tag categories, in order
tags_list = [
    ('gender', gender_tags),       
    ('rating', rating_classes),    
    ('hair color', hair_color_tags),   
    ('hair length', hair_length_tags),  
    ('hair style', hair_style_tags),   
    ('eye color', eye_color_tags),    
    ('eye misc.', eye_misc_tags),     
    ('expression', expression_tags),   
    ('breasts', breast_tags),       
    ('ass', ass_tags),          
    ('pose', pose_tags),         
    ('attire', attire_tags),       
]

N_CLASSES = sum([len(cat) for _, cat in tags_list])
DATASET_LENGTH = 301785

img_mean = np.load('./dataset_mean.npy')

def subset_accuracy_score(y_true, y_pred):
    differing_labels = K.sum(K.abs(y_true - K.round(y_pred)), axis=1)
    return K.mean(K.equal(differing_labels, 0))

def n_differing_labels(y_true, y_pred):
    differing_labels = K.sum(K.abs(y_true - K.round(y_pred)), axis=1)
    return K.mean(differing_labels)

def build_model():
    base_model = inception_v3.InceptionV3(weights=None, include_top=False, pooling='avg')

    x = base_model.output
    x = Dropout(0.20)(x)
    predictions = Dense(N_CLASSES, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights('./weights.027-0.4452.hdf5')

    return model

def main():
    model = build_model()
    
    with Image.open(sys.argv[1]) as img:
        resized = img.convert('RGB').resize((299, 299))
        arr = np.asarray(resized, np.uint8)
        arr = arr.astype(np.float32) / 255.0
        arr -= img_mean
        arr = (arr - 0.5) * 2.0 # models are trained with pixels in the range [-1, 1]
        arr = np.expand_dims(arr, 0)
        
        
        pred = np.squeeze(model.predict(arr, 1))
        predicted_labels = {}
        
    per_category_labels = []
    cur_idx = 0
    for cat_desc, cat in tags_list:
        cat_preds = np.zeros(len(cat))
        for i, tag in enumerate(cat):
            predicted_labels[tag] = pred[cur_idx+i]
            cat_preds[i] = pred[cur_idx+i]
            
        cat_labels = []
        top_labels = np.argsort(cat_preds)
        for i in top_labels[-3:]:
            cat_labels.insert(0, (cat[i], cat_preds[i]))
            
        per_category_labels.append(cat_labels)
        #if cat_preds[top_label] < 0.5:
        #    per_category_labels.append('none ({}?)'.format(cat[top_label]))
        #else:
        #    per_category_labels.append(cat[top_label])
            
        cur_idx += len(cat)
        
    print("\nTop Predicted Labels Per-Category:")
    for cat_idx, label_list in enumerate(per_category_labels):
        out_line = "{:14s} :".format(tags_list[cat_idx][0])
        
        for label, prob in label_list:
            out_line += " {:15s} ({:.3f}) |".format(label, prob)
            
        print(out_line)
    
    print("\nOverall Top Predicted Labels:")
    for tag in sorted(predicted_labels.keys(), key=lambda t: predicted_labels[t], reverse=True)[:50]:
        print("{:22s} : {:.3f}".format(tag, predicted_labels[tag]))

if __name__ == '__main__':
    main()

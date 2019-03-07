import pandas as pd
import math
import numpy as np
import tensorflow as tf
import sys
from PIL import Image
from pathlib import Path
import shutil

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
    (False, gender_tags),       
    (False, rating_classes),    
    (True, hair_color_tags),   
    (True, hair_length_tags),  
    (True, hair_style_tags),   
    (True, eye_color_tags),    
    (True, eye_misc_tags),     
    (True, expression_tags),   
    (False, breast_tags),       
    (False, ass_tags),          
    (False, pose_tags),         
    (False, attire_tags),       
]

flat_tags = []

for active, cat in tags_list:
    if active:
        flat_tags.extend(cat)

N_CLASSES = sum([len(cat) for _, cat in filter(lambda o: o[0], tags_list)]) 

def build_model():
    base_model = inception_v3.InceptionV3(weights=None, include_top=False, pooling='avg')

    x = base_model.output
    #x = Dropout(0.20)(x)
    predictions = Dense(N_CLASSES, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights('./tmp-weights/weights.010.hdf5')

    return model

def main():
    model = build_model()
    
    print("Loading metadata...")
    df = pd.read_csv('./2018-current.csv.gz', index_col=0)
    
    base_path = Path('G:/danbooru2018/original')
    
    false_negatives = 0
    false_positives = 0
    correct_tags = 0
    total_tags = 0
    
    for idx, row in df.iterrows():
        try:
            src = base_path / row[2]
            dst = Path('./images') / Path(row[2]).name
            
            shutil.copyfile(str(src), str(dst))
            
            with Image.open(src) as im:
                resized = im.convert('RGB').resize((299, 299))
                arr = np.asarray(resized, np.uint8)
                arr = arr.astype(np.float32) / 255.0
                #arr -= img_mean
                arr = (arr - 0.5) * 2.0
                arr = np.expand_dims(arr, 0)
            
                y_true = np.array(row[3:], np.float32)
                y_pred = np.squeeze(model.predict(arr, 1))
                
                print("Image {} ({}):".format(idx, row[2]))
                for tag, y_true_i, y_pred_i in zip(flat_tags, y_true, y_pred):
                    y_true_i = 1 if y_true_i >= 0.5 else 0
                    y_pred_i = 1 if y_pred_i >= 0.5 else 0
                    
                    if y_true_i > y_pred_i:
                        print("    False negative: `{}` - true={}, pred={}".format(tag, y_true_i, y_pred_i))
                        false_negatives += 1
                    elif y_true_i < y_pred_i:
                        print("    False positive: `{}` - true={}, pred={}".format(tag, y_true_i, y_pred_i))
                        false_positives += 1
                    else:
                        correct_tags += 1
                        
                    total_tags += 1
                    # elif y_true_i == 1:
                    #     print("    Correct tag `{}` - true={}, pred={}".format(tag, y_true_i, y_pred_i))
                        
        except OSError:
            pass
            
    
    print("Correct tags:    {} ({:6.3%})".format(correct_tags, correct_tags / total_tags))
    print("False positives: {} ({:6.3%})".format(false_positives, false_positives / total_tags))
    print("False negatives: {} ({:6.3%})".format(false_negatives, false_negatives / total_tags))
            
if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
import sys

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
    gender_tags,       
    rating_classes,    
    hair_color_tags,   
    hair_length_tags,  
    hair_style_tags,   
    eye_color_tags,    
    eye_misc_tags,     
    expression_tags,   
    breast_tags,       
    ass_tags,          
    pose_tags,         
    attire_tags,
]

all_tags = []
for l in tags_list:
    all_tags.extend(l)

N_CLASSES = sum([len(cat) for cat in tags_list])

def main():
    df = pd.read_csv(sys.argv[1], index_col=0)
    filtered = df.filter(items=all_tags)
    print(df.head())
    
    n_images = len(filtered)
    tag_sums = filtered.sum(axis=0, numeric_only=True)
    
    total_tag_count = np.sum(tag_sums.values)
    invalid_classes = []
    imrs = {}
    
    for idx, val in tag_sums.iteritems():
        if idx != 'id':
            if val < 1:
                invalid_classes.append(idx)
                continue
            
            n_neg = (n_images - val)
            imbalance_ratio = max(n_neg, val) / min(n_neg, val)
            imrs[idx] = imbalance_ratio
    
    imr_vals = list(imrs.values())
    std = np.std(imr_vals)
    
    filtered_imrs = []
    for idx, val in tag_sums.iteritems():
        if idx == 'id' or idx not in imrs:
            continue
            
        if imrs[idx] > (2*std):
            print("{:22s}: {:6d} images (IMR = {:.3f}) (outlier)".format(idx, val, imrs[idx]))
            continue
        
        filtered_imrs.append(imrs[idx])
        print("{:22s}: {:6d} images (IMR = {:.3f})".format(idx, val, imrs[idx]))
    
    for c in invalid_classes:
        print("Class does not have enough samples: {} ({} samples)".format(c, tag_sums[c]))
        
    print("Average IMR: {:.3f} (w/o outliers)".format(np.mean(filtered_imrs)))
    print("StdDev:      {:.3f}".format(std))
    print("Min IMR:     {:.3f}".format(np.amin(filtered_imrs)))
    print("Max IMR:     {:.3f}".format(np.amax(filtered_imrs)))
        
if __name__ == '__main__':
    main()

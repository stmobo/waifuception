import pandas as pd
import numpy as np

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

N_CLASSES = sum([len(cat) for cat in tags_list])

def main():
    df = pd.read_csv('./2018-current.csv.gz', index_col=0)
    
    n_images = len(df)
    tag_sums = df.sum(axis=0, numeric_only=True)
    
    total_tag_count = np.sum(tag_sums.values[1:])
    
    for idx, val in tag_sums.iteritems():
        if idx != 'id':
            print("{:22s}: {:6d} images ({:4.2%})".format(idx, val, val / total_tag_count))
    
    sub_idx = 0
    total_counts = np.zeros(N_CLASSES)
    for cat in tags_list:
        for idx, tag in enumerate(cat):
            total_counts[idx+sub_idx] = tag_sums[tag]
            
        sub_idx += len(cat)
    
    np.save('./total_tag_counts.npy', total_counts)
    
if __name__ == '__main__':
    main()

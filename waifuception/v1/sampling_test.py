import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import deque
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
    (True, attire_tags),
]

N_CLASSES = sum([len(cat) for _, cat in filter(lambda o: o[0], tags_list)]) 

all_classes = []
sampling_classes = []
for active, cls in tags_list:
    all_classes.extend(cls)
        
per_class_unused = {}
per_class_used = {}
total_tag_prevalence = {}
    
df = pd.read_csv('./2018-current-2.csv.gz', index_col=1)

for cls in all_classes:
    print("Loading: "+cls)
    ids = df.index[df[cls] == 1].tolist()
    
    with open('split_classes_2/'+cls+'.json', 'w', encoding='utf-8') as f:
        json.dump(ids, f)
        
    print("    > {} samples".format(len(ids)))
    
    sampling_classes.append(cls)
    per_class_unused[cls] = deque(ids)
    per_class_used[cls] = []
    total_tag_prevalence[cls] = len(ids)
    
prevalences = []
for _, taglist in tags_list:
    n = 0
    for tag in taglist:
        n += total_tag_prevalence[tag]
    
    prevalences.append(n)

TOTAL_LABEL_COUNT = sum(prevalences)
prevalences = (TOTAL_LABEL_COUNT - np.array(prevalences)) / TOTAL_LABEL_COUNT
print(prevalences)

sys.exit(0)
        
filtered_df = df.filter(sampling_classes)
print(filtered_df.head())
        
N_SAMPLED_CLASSES = len(sampling_classes)
print("Sampling from {} classes".format(N_SAMPLED_CLASSES))

def generate_random_batch(batch_sz=32):
    class_counts = np.zeros(N_SAMPLED_CLASSES)
    batch = []
    
    for i in range(batch_sz):
        # Find the indices of the classes which appear least:
        s = np.argsort(class_counts)
        min_class = s[np.where(class_counts[s] == class_counts[s[0]])] 
        random_cls = sampling_classes[np.random.choice(min_class)]

        random_id = per_class_unused[random_cls].popleft()
        per_class_used[random_cls].append(random_id)
        
        if len(per_class_unused[random_cls]) == 0:
            np.random.shuffle(per_class_used[random_cls])
            
            per_class_unused[random_cls] = deque(per_class_used[random_cls])
            per_class_used[random_cls] = []
        
        batch.append(random_id)
        labels = filtered_df.loc[random_id]
        
        class_counts += labels.values
        
    print("Class counts: ")
    for cls, count in sorted(zip(sampling_classes, class_counts), key=lambda o:o[1], reverse=True):
        print("{:22s}: {} ({:.1%} vs. {:.1%} overall prevalence)".format(
            cls, count,
            count / np.sum(class_counts),
            total_tag_prevalence[cls] / TOTAL_LABEL_COUNT
        ))
        
    return batch
    
if __name__ == '__main__':
    generate_random_batch()
        

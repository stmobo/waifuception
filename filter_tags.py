import ujson as json
import csv
import multiprocessing as mp
from pathlib import Path

gender_tags = ['1girl', '1boy']
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

tag_indices = {}
header_row = ['id', 'ext', 'image_width', 'image_height', 'source', 'character']

cur_idx = 0
for cat in tags_list:
    header_row.extend(cat)
    for subidx, tag in enumerate(cat):
        tag_indices[tag] = cur_idx + subidx
    
    cur_idx += len(cat)
    
tag_indices['s'] = tag_indices['safe']
tag_indices['q'] = tag_indices['questionable']
tag_indices['e'] = tag_indices['explicit']
    

def file_handler(file_idx):
    print("[file-{:02d}] Worker starting...".format(file_idx))
    main_fname = 'metadata/2018{:012d}'.format(file_idx)
    out_fname = 'preprocessed/2018-{:02d}.csv'.format(file_idx)
    
    with open(main_fname, 'r', encoding='utf-8') as in_f:
        with open(out_fname, 'w', encoding='utf-8', newline='') as out_f:
            writer = csv.writer(out_f)
            writer.writerow(header_row)
            
            n_output_rows = 0
            
            for line_no, line in enumerate(in_f):
                line_no += 1
                doc = json.loads(line)
                
                img_id = int(doc['id'])
                width = int(doc['image_width'])
                height = int(doc['image_height'])
                ext = doc['file_ext']
                
                if width <= 1 or height <= 1 or ext not in ['png', 'jpg', 'jpeg']:
                    continue
                
                is_solo = False
                classes_list = [0] * N_CLASSES
                
                character = ''
                source = ''
                
                classes_list[tag_indices[doc['rating']]] = 1
                
                for tag_doc in doc['tags']:
                    tag = tag_doc['name']
                    tag_category = int(tag_doc['category'])
                    # 0 = image content tag
                    # 1 = artist tag
                    # 2 = metadata tag?
                    # 3 = source material tag
                    # 4 = character tag
                    
                    if tag_category == 0: # Image content tag
                        if tag == 'solo':
                            is_solo = True
                        else:
                            try:
                                classes_list[tag_indices[tag]] = 1
                            except KeyError:
                                pass
                    elif tag_category == 3:
                        source = tag
                    elif tag_category == 4:
                        character = tag
                            
                if not is_solo or sum(classes_list) < 6:
                    continue
                
                writer.writerow([img_id, ext, width, height, source, character] + classes_list)
                n_output_rows += 1
                
                if line_no % 10000 == 0:
                    print("[file-{:02d}] Processed line {}, output {} rows after filtering".format(file_idx, line_no, n_output_rows))
            
            
def main():
    with mp.Pool(6) as p:
        p.map(file_handler, range(0, 17), 1)
        
        p.terminate()
        p.join()
    
if __name__ == '__main__':
    main()

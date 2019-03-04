#import keras
#from keras.applications import inception_v3
#from keras.models import Model
#from keras.layers import Dense, Dropout

hair_color_tags = ['aqua_hair', 'black_hair', 'blonde_hair', 'blue_hair', 'brown_hair', 'green_hair', 'grey_hair', 'orange_hair', 'pink_hair', 'purple_hair', 'red_hair', 'silver_hair', 'white_hair', 'multicolored_hair']
hair_length_tags = ['very_short_hair', 'short_hair', 'medium_hair', 'long_hair', 'very_long_hair', 'bald']
hair_style_tags = ['curly_hair', 'drill_hair', 'flipped_hair', 'hair_flaps', 'messy_hair', 'pointy_hair', 'ringlets', 'spiked_hair', 'wavy_hair', 'bangs', 'ahoge', 'braid', 'hair_bun', 'ponytail', 'twintails']

eye_color_tags = ['aqua_eyes', 'black_eyes', 'blue_eyes', 'brown_eyes', 'green_eyes', 'grey_eyes', 'orange_eyes', 'pink_eyes', 'purple_eyes', 'red_eyes', 'silver_eyes', 'white_eyes', 'yellow_eyes', 'heterochromia']
eye_misc_tags = ['closed_eyes', 'one_eye_closed', 'glasses']

expression_tags = ['angry', 'annoyed', 'blush', 'embarrassed', 'bored', 'confused', 'crazy', 'disdain', 'envy', 'expressionless', 'flustered', 'frustrated', 'happy', 'nervous', 'pout', 'sad', 'scared', 'panicking', 'worried', 'serious', 'sigh', 'sleepy', 'sulking', 'thinking', 'ahegao']

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
    hair_color_tags,    # 14 classes
    hair_length_tags,   # 6  classes
    hair_style_tags,    # 15 classes
    eye_color_tags,     # 14 classes
    eye_misc_tags,      # 3  classes
    expression_tags,    # 25 classes
    breast_tags,        # 6  classes
    ass_tags,           # 3  classes
    pose_tags,          # 4  classes
    attire_tags,        # 44 classes
]
# total: 134 classes

N_CLASSES = sum([len(cat) for cat in tags_list])

def build_model():
    inception = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')

    x = inception.output
    x = Dropout(0.20)(x)
    x = Dense(1024, activation='relu')(x)

    predictions = Dense(N_CLASSES, activation='sigmoid')(x)

    for layer in inception.layers:
        layer.trainable = False

    model = Model(inputs=inception.input, outputs=predictions)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    model.summary()

    return model

if __name__ == '__main__':
    for cat in tags_list:
        print("    "+str(len(cat))+" classes")
    print("total: "+str(N_CLASSES)+" classes")

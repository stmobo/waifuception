import json
import requests
import io
from PIL import Image
import sys
import traceback
from pathlib import Path


def get_post(post_id, output_dir):
    print("Grabbing metadata for post "+post_id+"...")
    meta_req = requests.get('https://danbooru.donmai.us/posts/{}.json'.format(post_id))
    metadata = meta_req.json()

    print("Grabbing image for post "+post_id+"...")
    img_req = requests.get(metadata['file_url'])
    bio = io.BytesIO()
    for chunk in img_req.iter_content(chunk_size=128):
        bio.write(chunk)
    bio.seek(0)

    with Image.open(bio) as img:
        img.convert('RGB').save(output_dir / '{}.jpg'.format(post_id))

    return metadata

ratings = {
    's': 'safe',
    'q': 'questionable',
    'e': 'explicit'
}
def process_metadata(meta):
    out = {}
    out['id'] = int(meta['id'])
    out['tags'] = meta['tag_string'].split(' ')
    out['rating'] = ratings[meta['rating']]

    return out

def main():
    output_dir = Path(sys.argv[1])
    post_meta = []

    for post in sys.argv[2:]:
        try:
            post_meta.append(process_metadata(get_post(post, output_dir)))
        except:
            traceback.print_exc()

    with open(str(output_dir / 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(post_meta, f)

if __name__ == '__main__':
    main()

]65;6003;1cfrom img2dataset import download
import shutil
import os
from clip_retrieval.clip_client import ClipClient, Modality
import numpy as np
import csv


def make_query_list(query_file, num_images=1000):

    animals = ["dog", "cat", "monkey", "elephant", "lion", "tiger", "zebra", "giraffe", "bear", "fox", "rabbit", "deer", "wolf", "horse", "cow", "sheep", "goat", "pig", "duck", "penguin", "panda", "koala", "kangaroo", "hippopotamus", "crocodile", "snake", "turtle", "frog"]
    verbs = ["walking", "running", "jumping", "swimming", "climbing", "crawling", "galloping", "leaping", "sneaking", "roaming", "napping", "playing", "sleeping", "yawning", "fighting", "jumping", "exploring", "observing"]

    query_list = [f"a photo of a {animal} {verb}" for animal in animals for verb in verbs]

    query_file = open('query.csv', 'w')
    client = ClipClient(url="https://knn.laion.ai/knn-service", indice_name="laion5B-L-14", num_images=num_images, )

    writer = csv.writer(query_file)
    writer.writerow(['URL', 'TEXT'])

    for query in query_list:
        results = client.query(text=query)

        for r in results:
            writer.writerow((r['url'], r['caption']))
        

def download_query_file(query_file, output_dir):
    download(
        processes_count=16,
        thread_count=32,
        url_list=query_file,
        resize_mode='center_crop',
        image_size=512,
        output_folder=output_dir,
        output_format="files",
        input_format="csv",
        url_col="URL",
        caption_col="TEXT",
        number_sample_per_shard=1000,
        distributor="multiprocessing",
        min_image_size=256,
    )
if __name__=="__main__":
    make_query_list('query.csv', 5000)
    download_query_file('query.csv', ' /mnt/disks/persist/dataset/')

import json
from typing import Iterator
import os
import requests


server_url = os.getenv("SERVER_URL", "http://15.237.95.219:5000")
ref_file = "part_1.mp3" #Reference audio to use to clone the speaker
output_path = "french_speaker3.json"

def get_speaker(ref_audio,server_url):
    files = {"wav_file": ("reference.wav", open(ref_audio, "rb"))}
    response = requests.post(f"{server_url}/clone_speaker", files=files)
    return response.json()



if __name__ == "__main__":

    print("Computing the latents for a new reference...")
    speaker = get_speaker(ref_file, server_url)
    with open(output_path, "w") as file:
        json.dump(speaker, file)

    

   

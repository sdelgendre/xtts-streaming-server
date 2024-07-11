import argparse
import json
import shutil
import subprocess
import sys
import time
from typing import Iterator
import os
import requests
import audioop

server_url = os.getenv("SERVER_URL", "http://localhost:8000")
speaker_file_path = "french_speaker3.json"
file_counter = 0
text = "Mon nom est Yoann, et je pense que c'est vraiment sympa de manger des crêpes, vous ne trouvez pas ?"

output_file = "./test_outputs/output_french"
while os.path.isfile(output_file+str(file_counter)+'.wav'):
    file_counter += 1
output_file = output_file+str(file_counter)+'.wav'


def convert_wav_chunk_to_ulaw_chunk(wav_chunk, sample_width=2): 
    # The sample_width parameter corresponds to the number of bytes used per sample, default is 2 for 16-bit audio
    
    if sample_width not in {1, 2, 4}:
        raise ValueError("sample_width must be 1, 2, or 4")
        
    # Convert the WAV audio chunk to u-Law encoding 
    try:
        ulaw_chunk = audioop.lin2ulaw(wav_chunk, sample_width) 
    except audioop.error as e:
        print(f"Error converting WAV chunk to u-Law: {e}")
        return None
    
    return ulaw_chunk

def is_installed(lib_name: str) -> bool:
    lib = shutil.which(lib_name)
    if lib is None:
        return False
    return True


def save(audio: bytes, filename: str) -> None:
    with open(filename, "wb") as f:
        f.write(audio)


def stream_ffplay(audio_stream, output_file, save=True):
    if not save:
        ffplay_cmd = ["ffplay", "-nodisp", "-probesize", "1024", "-autoexit", "-"]
    else:
        print("Saving to ", output_file)
        ffplay_cmd = ["ffmpeg", "-probesize", "1024", "-i", "-", output_file, '-ar', '8000']
        # ffplay_cmd = ["ffmpeg", "-probesize", "1024", "-i", '-','-c:a', 'pcm_mulaw', '-ar', '8000', output_file]

    ffplay_proc = subprocess.Popen(ffplay_cmd, stdin=subprocess.PIPE)
    for chunk in audio_stream:
        if chunk is not None:
            ffplay_proc.stdin.write(chunk)

    # close on finish
    ffplay_proc.stdin.close()
    ffplay_proc.wait()


def tts(text, speaker, language, server_url, stream_chunk_size) -> Iterator[bytes]:
    start = time.perf_counter()
    speaker["text"] = text
    speaker["language"] = language
    speaker["stream_chunk_size"] = stream_chunk_size  # you can reduce it to get faster response, but degrade quality
    res = requests.post(
        f"{server_url}/tts_stream/ulaw",
        json=speaker,
        stream=True,
    )
    end = time.perf_counter()
    print(f"Time to make POST: {end-start}s", file=sys.stderr)

    if res.status_code != 200:
        print("Error:", res.text)
        sys.exit(1)

    first = True
    for chunk in res.iter_content(chunk_size=512):
        if first:
            end = time.perf_counter()
            print(f"Time to first chunk: {end-start}s", file=sys.stderr)
            first = False
        if chunk:
            yield chunk

    print("⏱️ response.elapsed:", res.elapsed)


def get_speaker(ref_audio,server_url):
    files = {"wav_file": ("reference.wav", open(ref_audio, "rb"))}
    response = requests.post(f"{server_url}/clone_speaker", files=files)
    return response.json()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        default=text,
        help="text input for TTS"
    )
    parser.add_argument(
        "--language",
        default="fr",
        help="Language to use default is 'en'  (English)"
    )
    parser.add_argument(
        "--output_file",
        default=output_file,
        help="Save TTS output to given filename"
    )
    parser.add_argument(
        "--ref_file",
        default=None,
        help="Reference audio file to use, when not given will use default"
    )
    parser.add_argument(
        "--server_url",
        default=server_url,
        help="Server url http://localhost:8000 default, change to your server location "
    )
    parser.add_argument(
        "--stream_chunk_size",
        default="30",
        help="Stream chunk size , 20 default, reducing will get faster latency but may degrade quality"
    )
    args = parser.parse_args()

    with open(speaker_file_path, "r") as file:
        speaker = json.load(file)

    if args.ref_file is not None:
        print("Computing the latents for a new reference...")
        speaker = get_speaker(args.ref_file, args.server_url)

    audio = stream_ffplay(
        tts(
            args.text,
            speaker,
            args.language,
            args.server_url,
            args.stream_chunk_size
        ), 
        args.output_file,
        save=bool(args.output_file)
    )

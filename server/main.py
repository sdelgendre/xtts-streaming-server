import base64
import io
import os
import tempfile
import wave
import torch
import numpy as np
from typing import List
from pydantic import BaseModel
from pydub import AudioSegment
import ffmpeg
import logging
import json

from fastapi import FastAPI, UploadFile, Body
from fastapi.responses import StreamingResponse


from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager


# Setting up torch and device
torch.set_num_threads(int(os.environ.get("NUM_THREADS", os.cpu_count())))
device = torch.device("cuda" if os.environ.get("USE_CPU", "0") == "0" else "cpu")
if not torch.cuda.is_available() and device == "cuda":
    raise RuntimeError("CUDA device unavailable, please use Dockerfile.cpu instead.") 

# Setting up logger for debug purposes (logger.debug(message))
logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)

# Loading custom model
custom_model_path = "tts_model/"
if os.path.exists(custom_model_path) and os.path.isfile(custom_model_path + "/config.json"):
    model_path = custom_model_path
    print("Loading custom model from", model_path, flush=True)
else:
    print("Loading default model", flush=True)
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    print("Downloading XTTS Model:", model_name, flush=True)
    ModelManager().download_model(model_name)
    model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))
    print("XTTS Model downloaded", flush=True)

# Loading configuration
print("Loading XTTS", flush=True)
config = XttsConfig()
config.load_json(os.path.join(model_path, "config.json"))
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=model_path, eval=True, use_deepspeed=True if device == "cuda" else False)
model.to(device)
print("XTTS Loaded.", flush=True)

print("Running XTTS Server ...", flush=True)

##### Run fastapi #####
app = FastAPI(
    title="XTTS Streaming server",
    description="""XTTS Streaming server""",
    version="0.0.1",
    docs_url="/",
)


@app.post("/clone_speaker")
def predict_speaker(wav_file: UploadFile):
    """Compute conditioning inputs from reference audio file."""
    temp_audio_name = next(tempfile._get_candidate_names())
    with open(temp_audio_name, "wb") as temp, torch.inference_mode():
        temp.write(io.BytesIO(wav_file.file.read()).getbuffer())
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            temp_audio_name
        )
    return {
        "gpt_cond_latent": gpt_cond_latent.cpu().squeeze().half().tolist(),
        "speaker_embedding": speaker_embedding.cpu().squeeze().half().tolist(),
    }

@app.post("/add_speaker")
def add_new_speaker(wav_file : UploadFile, voice_id: str):
    """
    Clone a voice and save the voice embedding in a json file. Returns 
    """
    speaker = predict_speaker(wav_file=wav_file)
    voice_path = os.path.join('voices', voice_id)
    path_counter = 0
    while os.path.exists(voice_path+'.json'):
        voice_path+=str(path_counter)
        path_counter += 1
    voice_path += '.json'
    with open(voice_path,'w') as speaker_file:
        json.dump(speaker,speaker_file)
    return voice_path.strip('/')[1].strip('.')[0]

def postprocess(wav):
    """Post process the output waveform"""
    if isinstance(wav, list):
        wav = torch.cat(wav, dim=0)
    wav = wav.clone().detach().cpu().numpy()
    wav = wav[None, : int(wav.shape[0])]
    wav = np.clip(wav, -1, 1)
    wav = (wav * 32767).astype(np.int16)
    return wav


# TODO concatenante bytes
def convert_wav_chunk_to_ulaw(chunk, sample_rate=24000, sample_width=2, nchannels=1):
    """
    Convert wav bytes chunk to ulaw bytes chunk
    TODO : FINISH METHOD IMPLEMENTATION
    """
    # if (len(chunk)%sample_width*nchannels != 0):
    #     padding = sample_width*nchannels - len(chunk)%sample_width*nchannels
    #     chunk += b'\x00'*padding
    buffer = io.BytesIO()
    chunk_segment = AudioSegment(chunk, sample_width=sample_width,frame_rate=sample_rate,channels=nchannels)
    chunk_segment_ulaw = chunk_segment.export(format="wav",codec='pcm_mulaw',parameters=["-ar","8000"])
    data = ''
    for i,line in enumerate(chunk_segment_ulaw.readlines()):
        logger.debug(line[90:])
        if i == 0:
            yield line[90:]
        else:
            yield line
    return bytes(data)

def encode_audio_common(
    frame_input, encode_base64=True, sample_rate=24000, sample_width=2, channels=1
):
    """Return base64 encoded audio"""
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)
    

    wav_buf.seek(0)
    if encode_base64:
        b64_encoded = base64.b64encode(wav_buf.getbuffer()).decode("utf-8")
        return b64_encoded
    else:
        return wav_buf.read()
    
def create_ulaw_header(encode_base64=False):
    """
    Generate ulaw encoding header for ulaw-encoded wav files
    """
    process = (
        ffmpeg
        .input('pipe:0', format='s16le', ac=1, ar=24000)
        .output('pipe:1', acodec='pcm_mulaw', ar=8000)
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )

    # Write audio data to the pipe and read the WAV file from the output
    out, err = process.communicate(input=b'')
    wav_buf = io.BytesIO()
    wav_buf.write(out)
    wav_buf.seek(0)
    if encode_base64:
        b64_encoded = base64.b64encode(wav_buf.getbuffer()).decode("utf-8")
        return b64_encoded
    else:
        return wav_buf.read()
    

class StreamingInputs(BaseModel):
    """
    Utility class for stream inputs
    """
    voice_id : str = 'default_speaker'
    text: str
    language: str
    add_wav_header: bool = False
    stream_chunk_size: str = "20"


def predict_streaming_generator(parsed_input: dict = Body(...), ulaw : bool = True):
    """
    Generate stream chunk generator for tts streaming
    """
    voice_id = parsed_input.voice_id
    voice_path = os.path.join('voices',voice_id+'.json')
    if not os.path.exists(voice_path):
        print("Speaker file not found, using default voice")
        voice_path = 'voices/default_speaker.json'

    with open(voice_path,'r') as speaker_file:
        speaker = json.load(speaker_file)
    speaker_embedding = torch.tensor(speaker['speaker_embedding']).unsqueeze(0).unsqueeze(-1)
    gpt_cond_latent = torch.tensor(speaker["gpt_cond_latent"]).reshape((-1, 1024)).unsqueeze(0)
    text = parsed_input.text
    language = parsed_input.language

    stream_chunk_size = int(parsed_input.stream_chunk_size)
    add_wav_header = parsed_input.add_wav_header


    chunks = model.inference_stream(
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        stream_chunk_size=stream_chunk_size,
        enable_text_splitting=True
    )

    for i, chunk in enumerate(chunks):
        chunk = postprocess(chunk)
    
        # Création du header si on est au premier chunk
        if chunk is not None:
            if add_wav_header:
                chunk_with_header = encode_audio_common(chunk, encode_base64=False)
                yield chunk_with_header
            else:
                yield chunk.tobytes()



        # if i == 0 and add_wav_header:
        #     # Header
        #     yield encode_audio_common(chunk, encode_base64=False)
        # elif chunk is not None:
        #     if ulaw:
        #         chunk = convert_wav_chunk_to_ulaw(chunk.tobytes())
        #         yield chunk
        #     else:
        #         yield chunk.tobytes()

@app.post("/tts_stream")
def predict_streaming_endpoint(parsed_input: StreamingInputs):
    """
    Create stream tts endpoint
    """
    return StreamingResponse(
        predict_streaming_generator(parsed_input, ulaw=False),
        media_type="audio/wav",
    )

@app.post("/tts_stream/ulaw")
def predict_streaming_endpoint(parsed_input: StreamingInputs):
    """
    Create stream tts endpoint with ulaw conversion
    """
    return StreamingResponse(
        predict_streaming_generator(parsed_input, ulaw=True),
        media_type="audio/wav",
    )

class TTSInputs(BaseModel):
    """
    Utility class for TTS inputs
    """
    voice_id : str = 'default_speaker'
    text: str
    add_wav_header: bool = False
    language: str

@app.post("/tts")
def predict_speech(parsed_input: TTSInputs):
    """
    Generate tts speech
    """
    voice_id = parsed_input.voice_id
    print(os.curdir)
    voice_path = os.path.join('voices',voice_id+'.json')
    if not os.path.exists(voice_path):
        print("Speaker file not found, using default voice")
        voice_path = 'voices/default_speaker.json'
    
    with open(voice_path,'r') as speaker_file:
        speaker = json.load(speaker_file)
    speaker_embedding = torch.tensor(speaker["speaker_embedding"]).unsqueeze(0).unsqueeze(-1)
    gpt_cond_latent = torch.tensor(speaker["gpt_cond_latent"]).reshape((-1, 1024)).unsqueeze(0)
    text = parsed_input.text
    language = parsed_input.language
    add_wav_header = parsed_input.add_wav_header

    out = model.inference(
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
    )

    wav = postprocess(torch.tensor(out["wav"]))
    if add_wav_header:
        return encode_audio_common(wav.tobytes())
    else:
        return wav.tobytes()


@app.get("/studio_speakers")
def get_speakers():
    """
    Return speakers managed by the model manager
    """
    if hasattr(model, "speaker_manager") and hasattr(model.speaker_manager, "speakers"):
        return {
            speaker: {
                "speaker_embedding": model.speaker_manager.speakers[speaker]["speaker_embedding"].cpu().squeeze().half().tolist(),
                "gpt_cond_latent": model.speaker_manager.speakers[speaker]["gpt_cond_latent"].cpu().squeeze().half().tolist(),
            }
            for speaker in model.speaker_manager.speakers.keys()
        }
    else:
        return {}
        
@app.get("/languages")
def get_languages():
    """
    Return languages supported by the model
    """
    return config.languages


@app.get("/get_voice_ids")
def get_voice_ids():
    """
    Return voice available in the voice folder
    """
    try:
        # List all files and directories in the specified directory
        entries = os.listdir('voices')
        # Filter out directories, keeping only json files, return only the name
        files = [entry.split('.')[0] for entry in entries if os.path.isfile(os.path.join('voices', entry)) and entry.split('.')[1] == 'json']
        
        return files
    except Exception as e:
        print(f"An error occurred while checking voice ids: {e}")
        return []
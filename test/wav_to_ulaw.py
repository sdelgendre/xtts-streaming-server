from pydub import AudioSegment
def convert_ulaw_to_wav(input_file, output_file):
    with open(input_file, "rb") as f:
        ulaw_data = f.read()
    wav_data = AudioSegment( ulaw_data, sample_width=2, frame_rate=8000, channels=1)
    wav_data.export(output_file, format="wav")
    print(f"Converted {input_file} to {output_file}")
    
convert_ulaw_to_wav("nejzj3.ulaw", "output3.wav")
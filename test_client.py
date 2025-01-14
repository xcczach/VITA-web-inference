import requests
import soundfile as sf
import io
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=9234)
parser.add_argument("--zh-sample", action="store_true")
args = parser.parse_args()

url = f"http://localhost:{args.port}/vita"
audio_data, sample_rate = sf.read(
    "zh_sample.wav" if args.zh_sample else "en_sample.wav"
)
data = {"sample_rate": sample_rate, "audio_data": audio_data.tolist()}

response = requests.post(url, json=data, stream=True)

if response.status_code == 200:
    audio_chunks = []

    buffer = io.BytesIO()
    for chunk in response.iter_content(chunk_size=4096):
        buffer.write(chunk)

    buffer.seek(0)

    audio_data, sample_rate = sf.read(buffer)
    sf.write("output_audio.wav", audio_data, sample_rate)

    print("Audio saved at output_audio.wav")
else:
    print(f"Bad request：{response.status_code}")

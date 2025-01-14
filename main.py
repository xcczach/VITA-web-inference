import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from ml_web_inference import (
    expose,
    Request,
    StreamingResponse,
)
import torch
import io
import argparse
import time
import numpy as np
from PIL import Image
from vita.model.builder import load_pretrained_model
from decord import VideoReader, cpu
from vita.constants import (
    DEFAULT_AUDIO_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    MAX_IMAGE_LENGTH,
)
from vita.conversation import SeparatorStyle, conv_templates
from vita.model.builder import load_pretrained_model
from vita.util.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_audio_token,
    tokenizer_image_token,
)
from vita.util.utils import disable_torch_init
from vita.model.vita_tts.decoder.llm2tts import llm2TTS

from download import download_ckpt
import tempfile
import soundfile as sf
from vita.model.vita_tts.decoder.ticodec.vqvae_tester import VqvaeTester
import setproctitle
from vita.model.vita_tts.decoder.llm2tts import llm2TTS


tokenizer = None
model = None
context_len = None
audio_processor = None
tts = None
model_type = "qwen2p5_instruct"


async def inference(request: Request) -> StreamingResponse:
    data = await request.json()
    sample_rate = data["sample_rate"]
    audio_data = data["audio_data"]

    conv_mode = "qwen2p5_instruct"
    temperature = 0.01
    top_p = None
    num_beams = 1

    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio_file:
        temp_audio_path = temp_audio_file.name
        sf.write(temp_audio_path, data=audio_data, samplerate=sample_rate)
        audio, audio_for_llm_lens = audio_processor.process(
            os.path.join(temp_audio_path)
        )
    device = model.device
    audio_length = audio.shape[0]
    audio = torch.unsqueeze(audio, dim=0)
    audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
    audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
    audios = dict()
    audios["audios"] = audio.half().cuda(device)
    audios["lengths"] = audio_length.half().cuda(device)
    audios["lengths_for_llm"] = audio_for_llm_lens.cuda(device)

    qs = DEFAULT_AUDIO_TOKEN
    image_tensor = torch.zeros((1, 3, 448, 448)).to(
        dtype=model.dtype, device=model.device
    )
    modality = "lang"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt(modality)

    input_ids = (
        tokenizer_image_audio_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .cuda(device)
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            audios=audios,
            do_sample=False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            shared_v_pid_stride=None,  # 2#16#8#4#1#None,
        )
    output_ids = output_ids.sequences
    input_token_len = input_ids.shape[1]
    if model_type == "mixtral-8x7b":
        n_diff_input_output = (
            (input_ids != output_ids[:, :input_token_len]).sum().item()
        )
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
            output_ids = output_ids[:, input_token_len:]
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    print(outputs)

    result = io.BytesIO()
    # torchaudio.save(result, result_arr, target_sample_rate, format="wav")
    result.seek(0)
    return StreamingResponse(result, media_type="application/octet-stream")


def init():
    global tokenizer, model, context_len, audio_processor, tts
    model_path = download_ckpt()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, _, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        model_type=model_type,
    )
    model.resize_token_embeddings(len(tokenizer))
    audio_encoder = model.get_audio_encoder()
    audio_encoder.to(dtype=torch.float16)
    audio_processor = audio_encoder.audio_processor
    model.eval()
    tts = llm2TTS(os.path.join(model_path, "vita_tts_ckpt/"))


def hangup():
    global audio_tokenizer, generator
    del audio_tokenizer
    del generator
    torch.cuda.empty_cache()


if __name__ == "__main__":
    setproctitle.setproctitle("vita-web-inference")
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9234)
    parser.add_argument("--api-name", type=str, default="vita")
    parser.add_argument("--hangup-timeout-sec", type=int, default=900)
    parser.add_argument("--hangup-interval-sec", type=int, default=60)
    args = parser.parse_args()
    expose(
        args.api_name,
        inference,
        port=args.port,
        hangup_timeout_sec=args.hangup_timeout_sec,
        hangup_interval_sec=args.hangup_interval_sec,
        init_function=init,
        hangup_function=hangup,
    )

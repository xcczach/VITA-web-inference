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
from download import download_ckpt
import tempfile
import soundfile as sf
from vita.model.vita_tts.decoder.ticodec.vqvae_tester import VqvaeTester
import setproctitle


tokenizer = None
model = None
context_len = None
audio_processor = None
tts = None


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
    if args.model_type == "mixtral-8x7b":
        n_diff_input_output = (
            (input_ids != output_ids[:, :input_token_len]).sum().item()
        )
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
            output_ids = output_ids[:, input_token_len:]
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]

    result = io.BytesIO()
    torchaudio.save(result, result_arr, target_sample_rate, format="wav")
    result.seek(0)
    return StreamingResponse(result, media_type="application/octet-stream")


def init():
    global tokenizer, model, context_len, audio_processor, tts
    model_path = download_ckpt()
    model_name = get_model_name_from_path(model_path)
    model_type = "qwen2p5_instruct"
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


class llm2TTS:
    def __init__(self, model_path):
        self.model = (
            self.get_model(model_path)
            .cuda()
            .to(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32)
        )
        self.infer = self.model.infer

        self.codec_model = VqvaeTester(
            config_path=model_path + "/codec/model.json",
            model_path=model_path + "/codec/final.pt",
            sample_rate=24000,
        )
        self.codec_model = self.codec_model.cuda()
        self.codec_model.vqvae.generator.remove_weight_norm()
        self.codec_model.vqvae.encoder.remove_weight_norm()
        self.codec_model.eval()

    def get_model_conf(self, model_path):
        model_conf = model_path + "/decoder/model.json"
        with open(model_conf, "rb") as f:
            print("reading a config file from " + model_conf)
            confs = json.load(f)
        # for asr, tts, mt
        idim, odim, args = confs
        return argparse.Namespace(**args)

    def get_model(self, model_path):
        args_load = self.get_model_conf(model_path)
        args_load = vars(args_load)
        args = argparse.Namespace(**args_load)
        odim = args.odim
        idim = args.idim
        model = LLM2TTSCodecAR(idim, odim, args)

        # Resume from a snapshot
        snapshot_dict = torch.load(
            model_path + "/decoder/final.pt", map_location=lambda storage, loc: storage
        )
        if "model" in snapshot_dict.keys():
            resume_model_dict = snapshot_dict["model"]
        else:
            resume_model_dict = snapshot_dict

        model_dict = model.state_dict()
        for key in model_dict.keys():
            if key in resume_model_dict.keys():
                if model_dict[key].shape == resume_model_dict[key].shape:
                    model_dict[key] = resume_model_dict[key]
                else:
                    print(
                        "Key {} has different shape, {} VS {}".format(
                            key, model_dict[key].shape, resume_model_dict[key].shape
                        )
                    )
            else:
                print("Key {} has not in resume model".format(key))
        model.load_state_dict(model_dict)
        model.eval()
        return model

    def find_min_sum_index(self, buffer, syn, N, threshold):
        """
        Find the index with the minimum sum of a sliding window in the given audio segment
        and perform operations based on this index.

        Parameters:
        - buffer (torch.Tensor): The buffer containing previously processed audio segments.
        - syn (torch.Tensor): The current audio segment to be processed.
        - N (int): The size of the sliding window used to calculate the sum.
        - threshold (float): Threshold value to determine whether to concatenate buffer and current segment or not.

        Returns:
        - tuple: A tuple containing the updated buffer and the processed audio segment.

        """
        arr = syn[0, 0, :]
        L = len(arr)
        mid = L // 2

        kernel = torch.ones(N).to(arr.device)
        window_sums = torch.nn.functional.conv1d(
            arr.abs().view(1, 1, -1), kernel.view(1, 1, -1), padding=0
        ).squeeze()

        start_index = mid - (N // 2)
        min_sum, min_index = torch.min(window_sums[start_index:], dim=0)

        # get the start and end index of the window
        start_index = max(0, min_index.item() + start_index)
        end_index = min(L, min_index.item() + N + start_index)

        # calculate the real min_sum and min_index
        min_sum_real, min_index_real = torch.min(
            arr[start_index:end_index].abs(), dim=0
        )
        min_index = min_index_real.item() + start_index

        min_sum = min_sum / N
        syn_clone = syn.clone()

        if min_sum < threshold:
            syn = torch.cat([buffer.clone(), syn[:, :, :min_index]], dim=-1)
            buffer = syn_clone[:, :, min_index:]
        else:
            buffer = torch.cat([buffer, syn_clone], dim=-1)
            syn = None
        return buffer, syn

    def run(
        self,
        hidden,
        top_k,
        prefix,
        codec_chunk_size=40,
        codec_padding_size=10,
        penalty_window_size=-1,
        penalty=1.1,
        N=2401,
        seg_threshold=0.01,
    ):
        """
        Run the speech decoder process.

        Parameters:
        - hidden (torch.Tensor): The output for embedding layer of the language model.
        - top_k (int): The number of top-k tokens to consider during inference.
        - prefix (str, optional): The hidden state from the language model.
        - codec_chunk_size (int, default=40): The size of each chunk to process in the codec model.
        - codec_padding_size (int, default=10): The amount of padding to add on each side of the codec chunk.
        - penalty_window_size (int, default=20): The window size for applying penalties during decoding.
        - penalty (float, default=1.1): The penalty factor.

        Yields:
        - torch.Tensor: Intermediate audio segments generated by the codec model.

        """
        codec_upsample_rate = 600
        left_padding = 0
        right_padding = codec_padding_size
        prefix = None
        buffer = torch.zeros([1, 1, 0]).to(hidden.device)
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=(
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
                ),
            ):
                print("Starting TTS...")
                token = torch.full(
                    (1, 0),
                    self.model.vocab_size,
                    dtype=torch.long,
                    device=hidden.device,
                )
                for next_token_id in self.infer(
                    hidden, top_k, prefix, penalty_window_size, penalty
                ):
                    token = torch.cat([token, next_token_id], dim=-1)
                    if token.size(1) == left_padding + codec_chunk_size + right_padding:
                        syn = self.codec_model.vqvae(
                            token.unsqueeze(-1),
                            torch.tensor(
                                self.codec_model.vqvae.h.global_tokens,
                                device=token.device,
                            )
                            .unsqueeze(0)
                            .unsqueeze(0),
                        )
                        print("Codec Done")
                        syn = syn[
                            :,
                            :,
                            left_padding
                            * codec_upsample_rate : -right_padding
                            * codec_upsample_rate,
                        ]
                        left_padding = codec_padding_size
                        token = token[:, -(left_padding + right_padding) :]
                        buffer, syn = self.find_min_sum_index(
                            buffer, syn, N, seg_threshold
                        )
                        if syn is not None:
                            yield syn
                if token.size(1) > 0:
                    print("Codec Done")
                    syn = self.codec_model.vqvae(
                        token.unsqueeze(-1),
                        torch.tensor(
                            self.codec_model.vqvae.h.global_tokens, device=token.device
                        )
                        .unsqueeze(0)
                        .unsqueeze(0),
                    )
                    syn = syn[:, :, left_padding * codec_upsample_rate :]
                    yield torch.cat([buffer, syn], dim=-1)


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

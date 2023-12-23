# parts of this looted from https://github.com/pythongosssss/ComfyUI-WD14-Tagger
import asyncio
import base64
import os
import re
import time
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

import comfy.utils
import folder_paths

model_fmt = ".gguf"
model_type = "llama"
system_message = (
    "You are an assistant who describes the content and composition of images. "
    "Describe only what you see in the image, not what you think the image is about. "
    "Be factual and literal. Do not use metaphors or similes. Be concise."
)


defaults = {
    "model": "llava-v1.5-7b-Q4_K",
    "mmproj": "llava-v1.5-7b-mmproj-Q4_0",
    "temperature": 0.2,
    "max_tokens": 40,
    "prompt": "Please describe this image in 10 to 20 words.",
    "n_gpu_layers": -1,
}


def get_ext_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def get_installed_models(mm_proj=False):
    if model_type not in folder_paths.folder_names_and_paths:
        models_dir = get_ext_dir("models", mkdir=True)
        folder_paths.add_model_folder_path(model_type, models_dir)

    models = folder_paths.get_filename_list(model_type)
    return [
        re.sub(rf"{model_fmt}$", "", m)
        for m in models
        if m.endswith(model_fmt) and ("mmproj" in m) == mm_proj
    ]


async def get_llava(
    model,
    mm_proj,
    n_gpu_layers=0,
):
    if n_gpu_layers is None:
        n_gpu_layers = 0

    assert isinstance(model, str), f"{model} {type(model)=}"
    assert isinstance(mm_proj, str), f"{mm_proj} {type(mm_proj)=}"
    assert isinstance(n_gpu_layers, int), f"{n_gpu_layers} {type(n_gpu_layers)=}"

    model_path = folder_paths.get_full_path(model_type, model + model_fmt)
    mmproj_path = folder_paths.get_full_path(model_type, mm_proj + model_fmt)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_path} does not exist")

    if not os.path.exists(mmproj_path):
        raise FileNotFoundError(f"Model {mmproj_path} does not exist")

    chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)

    start = time.monotonic()

    # noinspection PyTypeChecker
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        chat_format="llava-1-5",
        chat_handler=chat_handler,
        n_ctx=2048,  # n_ctx should be increased to accomodate the image embedding
        logits_all=True,
        verbose=False,
    )
    print(f"LLM loaded in {time.monotonic() - start:.1f}s")
    return llm


def encode(image: Image.Image):
    assert isinstance(image, Image.Image), f"{image} {type(image)}"
    with BytesIO() as output:
        image.save(output, format="PNG")
        image_bytes = output.getvalue()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:image/png;base64,{base64_image}"
    return image_url


async def get_caption(
    llm: Llama,
    image: Image.Image,
    prompt,
    temp,
    max_tokens=35,
):
    assert isinstance(image, Image.Image), f"{image} {type(image)=}"
    assert isinstance(system_message, str), f"{system_message} {type(system_message)=}"
    assert isinstance(prompt, str), f"{prompt} {type(prompt)=}"
    assert isinstance(temp, float), f"{temp} {type(temp)=}"
    assert isinstance(max_tokens, int), f"{max_tokens} {type(max_tokens)=}"

    file_url = encode(image)
    messages = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": file_url}},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    start = time.monotonic()
    response = llm.create_chat_completion(
        messages=messages,
        temperature=temp,
        max_tokens=max_tokens,
    )
    print(f"Response in {time.monotonic() - start:.1f}s")

    first_resp: dict = response["choices"][0]
    content = first_resp["message"]["content"]  # oh leave me alone type inferencing

    # print(json.dumps(messages, indent=2))
    # print(json.dumps(response, indent=2))
    # print(content)

    return content.strip()


def wait_for_async(async_fn, loop=None):
    res = []

    async def run_async():
        r = await async_fn()
        res.append(r)

    if loop is None:
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

    loop.run_until_complete(run_async())

    return res[0]


class LlavaCaptioner:
    @classmethod
    def INPUT_TYPES(s):
        all_models = get_installed_models()
        all_mmproj = get_installed_models(mm_proj=True)

        return {
            "required": {
                "image": ("IMAGE",),
                "model": (all_models,),
                "mm_proj": (all_mmproj,),
                "prompt": (
                    "STRING",
                    {"default": defaults["prompt"], "multiline": True},
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": defaults["max_tokens"],
                        "min": 0,
                        "max": 200,
                        "step": 5,
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": defaults["temperature"],
                        "min": 0.0,
                        "max": 1,
                        "step": 0.1,
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "caption"
    OUTPUT_NODE = True

    CATEGORY = "image"

    def caption(self, image, model, mm_proj, prompt, max_tokens, temperature):
        assert isinstance(image, torch.Tensor), f"{image} {type(image)=}"
        assert isinstance(model, str), f"{model} {type(model)=}"
        assert isinstance(mm_proj, str), f"{mm_proj} {type(mm_proj)=}"
        assert isinstance(prompt, str), f"{prompt} {type(prompt)=}"
        assert isinstance(max_tokens, int), f"{max_tokens} {type(max_tokens)=}"
        assert isinstance(temperature, float), f"{temperature} {type(temperature)=}"

        tensor = image * 255
        tensor = np.array(tensor, dtype=np.uint8)

        pbar = comfy.utils.ProgressBar(tensor.shape[0] + 1)

        llava = wait_for_async(lambda: get_llava(model, mm_proj, -1))
        pbar.update(1)

        tags = []
        for i in range(tensor.shape[0]):
            image = Image.fromarray(tensor[i])
            tags.append(
                wait_for_async(
                    lambda: get_caption(
                        llava,
                        image,
                        prompt,
                        temperature,
                        max_tokens,
                    )
                )
            )
            pbar.update(1)
        result = "\n".join(tags)
        return {"ui": {"tags": tags}, "result": (result,)}


NODE_CLASS_MAPPINGS = {
    "LLaVA Captioner //cd": LlavaCaptioner,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LLaVA Captioner //cd": "LLaVA Captioner 🌊",
}

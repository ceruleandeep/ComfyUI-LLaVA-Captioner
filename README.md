# ComfyUI LLaVA Captioner

A [ComfyUI](https://github.com/comfyanonymous/ComfyUI) extension for chatting with your images.


## Installation
1. `git clone https://github.com/ceruleandeep/ComfyUI-LLaVA-Captioner` into the `custom_nodes` folder 
    - e.g. `custom_nodes/ComfyUI-LLaVA-Captioner`  
2. Open a Command Prompt/Terminal/etc
3. Change to the `custom_nodes/ComfyUI-LLaVA-Captioner` folder you just created 
    - e.g. `cd C:\ComfyUI_windows_portable\ComfyUI\custom_nodes/ComfyUI-LLaVA-Captioner` or wherever you have it installed
4. Run `python install.py`
5. Download models from [🤗](https://huggingface.co/jartine/llava-v1.5-7B-GGUF/tree/main) into `models/llama`:
    - [llava-v1.5-7b-Q4_K.gguf](https://huggingface.co/jartine/llava-v1.5-7B-GGUF/resolve/main/llava-v1.5-7b-Q4_K.gguf)
    - [llava-v1.5-7b-mmproj-Q4_0.gguf](https://huggingface.co/jartine/llava-v1.5-7B-GGUF/resolve/main/llava-v1.5-7b-mmproj-Q4_0.gguf)

## Usage
Add the node via `image` -> `LLaVA Captioner //cd`  

Supports tagging and outputting multiple batched inputs.  
- **model**: The multimodal LLM model to use. People are most familiar with LLaVA but there's also [Obsidian](https://huggingface.co/nisten/obsidian-3b-multimodal-q6-gguf/tree/main) or [BakLLaVA](https://huggingface.co/mys/ggml_bakllava-1/tree/main) or [ShareGPT4](https://huggingface.co/Galunid/ShareGPT4V-gguf/tree/main) 
- **mmproj**: The multimodal projection that goes with the model
- **prompt**: Question to ask the LLM
- **max_tokens** Maximum length of response, in tokens. A token is approximately half a word.
- **temperature** How much randomness to allow in the result. While a lot of people are using the text-only Llama series models with temperatures up around 0.7 and enjoying the creativity, LLaVA's accuracy seems to benefit greatly from temperatures less than 0.2.


## Requirements
* [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python)

This is easy to install but getting it to use the GPU can be a bit of a saga.

GPU inference time is 4 secs per image on a RTX 4090 with 4GB of VRAM to spare, and 8 secs per image on a Macbook Pro M1. 
CPU inference time is 25 secs per image. If your inference times are closer to 25 than to 5, you're probably doing CPU inference.

Unfortunately the multimodal models in the Llama family need about a 4x larger context size than the text-only ones,
so the `llama.cpp` promise of everyone doing fast LLM inference on their CPUs hasn't quite arrived yet. If you have a GPU, put it to work.

## See also
* 
* [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
* [llama.cpp](https://github.com/ggerganov/llama.cpp)
* [ComfyUI-WD14-Tagger](https://github.com/pythongosssss/ComfyUI-WD14-Tagger)

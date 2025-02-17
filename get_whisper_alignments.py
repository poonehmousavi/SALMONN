# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import librosa
import json
import pickle



# LibriSQA_path = "path/LibriSQA-PartI_LibriSQA-PartI-test.json"
# IEMOCAP_path= "path/test.json"
# data_root = "path/IEMOCAP/IEMOCAP_full_release/"

import h5py
import tqdm


def move_to_cuda(input_dict):
    return {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in input_dict.items()}


def extract_whisper_LibriSQA(data_path, h5_filename, data_root, pipe, processor_qwen, model_qwen, processor_whisper):
    all_results = {}
    #with h5py.File(h5_filename, "a") as h5file:  # Open file in append mode
    with open(data_path, "r") as file:
        data = json.load(file)
        for i in tqdm.tqdm(data):
            wav_path = data_root + i["speech_path"].replace(".wav", ".flac")
            transcript = i["text"]
            prompt = i["question"]+". Answer the  question as short as possible (in 10-15 words)"
            uid = i["speech_path"].split('/')[-1].split('.')[0]

            result = pipe(wav_path, return_timestamps='word') # generate_kwargs={"return_timestamps": True})
            
            # get the 
            conversation = [
                {'role': 'system', 'content': 'You are a helpful assistant.'}, 
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": None},
                    {"type": "text", "text":  "Audio Transcription: {text}\n\n{question}".format(text=transcript, question=prompt.strip())},
                ]},
            ]
            prompt_text = processor_qwen.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

            result_uid = {'transcript': transcript, 
                          'whisper_results': result,
                          'qwen_prompt': prompt_text}
            
            print("Processing UID:", uid)

            # Ensure we don't overwrite existing data for the UID
            # Create a group for this UID
            all_results[uid] = result_uid
            pickle.dump(all_results, open(h5_filename, 'wb'))


            # inputs = processor_qwen(text=prompt_text, audios= [[]], return_tensors="pt", padding=True)
            # inputs = move_to_cuda(inputs)

            # outputs = model_qwen.generate(
            #     **inputs,
            #     max_new_tokens=150,
            #     return_dict_in_generate=True,
            #     output_hidden_states=True,
            # )
            # generate_ids = outputs.sequences[:, inputs['input_ids'].size(1):]
            # response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            # response_text_text, hidden_states_text = response, outputs.hidden_states

            #generate_outputs(uid,wav_path,transcript,prompt,h5file)

def extract_IEMOCAP(data_path,h5_filename,data_root):
    prompt = "classify the emotion of the speaker in only one word including anger, happiness, sadness, neutrality."
    # Step 1: Load JSON data from the file
    with h5py.File(h5_filename, "a") as h5file:  # Open file in append mode
        with open(data_path, "r") as file:
            json_data = json.load(file)

        # Step 2: Update the `wav` paths
        for key, value in json_data.items():
            wav_path = value["wav"].replace("{data_root}", data_root)
            transcript =value["text"]
            generate_outputs(key,wav_path,transcript,prompt,h5file)


import argparse

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Run tasks based on the given input.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
            "--options",
            nargs="+",
            help="override some settings in the used config, the key-value pair "
            "in xxx=yyy format will be merged into config file (deprecate), "
            "change to --cfg-options instead.",
        )

    # Define arguments
    parser.add_argument(
        "--task",
        type=str,
        default="SQA",
        # required=True,
        choices=["SQA", "ER"],
        help="Task to perform. Options: 'SQA' for LibriSQA extraction, 'EER' for IEMOCAP extraction."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default='.',
        # required=True,
        help="Path to the dataset for the selected task."
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default='.',
        required=True,
        help="Path to output."
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default='.',
        required=True,
        help="data root where the audio file is saves"
    )
    # Parse arguments
    args = parser.parse_args()

    # set-up whisper
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    processor_qwen = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model_qwen = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")

    # Run task based on the input
    if args.task == "SQA":
        extract_whisper_LibriSQA(args.input_path, args.output_path, args.data_root, pipe, processor_qwen, model_qwen, processor)
    elif args.task == "ER":
        extract_whisper_IEMOCAP(args.input_path,args.output_path,args.data_root)


# python get_qwen_hiddenstates.py --task SQA (or ER)  --input_path {path_to-json file} --output_path {path-to_output.h5} --data_root {path_ _to_audio_file}

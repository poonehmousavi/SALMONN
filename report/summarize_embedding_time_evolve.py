import h5py
import torch
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import argparse
<<<<<<< HEAD
import math
import matplotlib.pyplot as plt
=======
>>>>>>> 0dc86209a5acc08195be69b9f3664c54f621c9c5

def norm_and_convolve(seq1, seq2):
    seq1, seq2 = seq1.squeeze(), seq2.squeeze()

    len1 = seq1.shape[0]
    len2 = seq2.shape[0]
    
    if len2 > len1: 
        longerseq = seq2
        shorterseq = seq1
    else:
        longerseq = seq1
        shorterseq = seq2

    norm = (longerseq ** 2).sum(-1, keepdim=True).sqrt()
    longerseq = longerseq / norm
    
    norm = (shorterseq ** 2).sum(-1, keepdim=True).sqrt()
    shorterseq = shorterseq / norm

<<<<<<< HEAD
    innerprods = (shorterseq.unsqueeze(0) * longerseq.unsqueeze(1)).sum(-1).cpu()
    return innerprods

def find_path(mat):
    for t in range(mat.shape[1]):
        pass

def convert_whisper(whisper_data, simmat):

    whisper_mat = torch.zeros(simmat.shape)
    chunks = whisper_data['whisper_results']['chunks']

    max_time = chunks[-1]['timestamp'][0]
    ntime_steps = simmat.shape[0]
    for t, chunk in enumerate(chunks):
        begin = math.floor((chunk['timestamp'][0] / max_time) * ntime_steps)

        if chunk['timestamp'][1] is not None:
            end = math.ceil((chunk['timestamp'][1] / max_time) * ntime_steps)
        else:
            end = simmat.shape[0]
        whisper_mat[begin:end, t] = 1
    return whisper_mat

=======
    innerprods = (shorterseq.unsqueeze(0) * longerseq.unsqueeze(1)).sum(-1)
    return innerprods

>>>>>>> 0dc86209a5acc08195be69b9f3664c54f621c9c5
if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Run tasks based on the given input.")
    parser.add_argument("--input_path", 
                        default="/path/QWEN-LibriSQA-hidden.h5",
                        type=str,  help='path to configuration file')

    parser.add_argument(
        "--output_path",
        type=str,
        default="/path/librisqa/qwen/librisqa_sumarized_data.pkl",
        # required=True,
        help="Path to the dataset for the selected task."
    )

<<<<<<< HEAD
    parser.add_argument(
        "--whisper_path",
        type=str,
        default=".",
        # required=True,
        help="Path to the dataset for the selected task."
    )

=======
>>>>>>> 0dc86209a5acc08195be69b9f3664c54f621c9c5
    parser.add_argument("--hidden_dim", default=4096, type=int, help='path to configuration file. 4096 for qwen and 5120 for salmon')
    # Parse arguments
    args = parser.parse_args()
# Load pre-trained sentence embedding model
    hidden_dim= args.hidden_dim
    text_embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    # Similarity Threshold (adjustable)
    SIMILARITY_THRESHOLD = 0.7  # Only keep pairs with >70% similarity

    # Load the HDF5 file
    h5_filename = args.input_path

    # Store filtered pairs
    filtered_data = {}
<<<<<<< HEAD
    skip_count = 0
    count = 0

    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    processor_qwen = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model_qwen = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")
    
    whisper_data = pickle.load(open(args.whisper_path, 'rb'))
=======
    skip_count=0
    count=0
>>>>>>> 0dc86209a5acc08195be69b9f3664c54f621c9c5
    with h5py.File(h5_filename, "r") as h5file:
        for UID in h5file.keys():
            count +=1
            print(f"Processing UID: {UID}")

            # Load response texts
            #response_text_audio = h5file[UID]["audio_model"].attrs["response_text"][0]  #change to input_audio_model if you want to get input hiddensates and audio_model if you want to get output hiddenstates
            #response_text_text = h5file[UID]["text_model"].attrs["response_text"][0] #change to input_text_model if you want to get input hiddensates and text_model if you want to get output hiddenstate

            response_text_audio = h5file[UID]["audio_model"].attrs["response_text"]  #change to input_audio_model if you want to get input hiddensates and audio_model if you want to get output hiddenstates
            response_text_text = h5file[UID]["text_model"].attrs["response_text"] #change to input_text_model if you want to get input hiddensates and text_model if you want to get output hiddenstate


            # Convert response texts into sentence embeddings
            audio_text_embedding = text_embedder.encode(response_text_audio)
            text_text_embedding = text_embedder.encode(response_text_text)

            # Compute Cosine Similarity of response texts
            text_similarity = cosine_similarity([audio_text_embedding], [text_text_embedding])[0, 0]
            print(f"Text Similarity for {UID}: {text_similarity:.2f}")

            # **Filter by similarity threshold**
            if text_similarity >= SIMILARITY_THRESHOLD:
                print(f"Keeping {UID} (Similarity: {text_similarity:.2f})")

                # Load embeddings
                audio_group = h5file[UID]["audio_model"]
                text_group = h5file[UID]["text_model"]

                # **Extract Correct Number of Layers**
                num_layers_audio = len(audio_group["tuple_0"])  # Number of layers
                num_layers_text = len(text_group["tuple_0"])

                assert num_layers_audio == num_layers_text, "Mismatch in number of layers!"

                # **Extract Sequence Lengths**
                seq_len_audio = len(audio_group)  # Variable sequence length
                seq_len_text = len(text_group)

                # Initialize storage for layer-wise embeddings (Excluding layer 0)
                audio_embeddings = np.zeros((seq_len_audio, num_layers_audio, hidden_dim))
                text_embeddings = np.zeros((seq_len_text, num_layers_text, hidden_dim))

                # **Extract All Layers but **
                # for seq_idx in range(0,seq_len_audio):
                #     for layer_idx in range(0, num_layers_audio):

                #         layer_group = audio_group[f"tuple_{seq_idx}"]
                #         
                #         # Extract and fix sequence dimension
                #         tensor = torch.tensor(layer_group[f"tensor_{layer_idx}"][:])  # (1, 1, 5120) or (1, input_seq. 5120) if it is seq-0 since it contains input hidden states.

                #         audio_embeddings[seq_idx, layer_idx, :] = tensor.numpy()  
                # only taking the input 
                
                all_innerprods = []
                for nlayer in range(num_layers_audio):
                    audio_embeddings = audio_group["tuple_0"][f'tensor_{nlayer}'][:]
                    text_embeddings = text_group["tuple_0"][f'tensor_{nlayer}'][:]

<<<<<<< HEAD
                    audio_embeddings = torch.from_numpy(audio_embeddings).cuda()
                    text_embeddings = torch.from_numpy(text_embeddings).cuda()

                    all_innerprods.append(norm_and_convolve(audio_embeddings, text_embeddings))

                
                whisper_uid = whisper_data[UID]

                text_whisper = whisper_uid['transcript']
                
                #qwen_tokens = processor_qwen.tokenizer(text_whisper)['input_ids']

                # take the last layer, and eliminate the prompt part
                promptpre = 27
                promptpost = -23
 
                qwen_tokens = processor_qwen.tokenizer(whisper_uid['qwen_prompt'])
                
                # last_layer = all_innerprods[-1][promptpre:promptpost, promptpre:promptpost]
                # whisper_mat = convert_whisper(whisper_uid, last_layer)
                # plt.imshow(whisper_mat.transpose(1, 0).cpu())
                # plt.savefig('whisper_mat.png')

                # plt.imshow(last_layer.transpose(1, 0))
                # plt.savefig('last_layer.png')




                
                


=======
                    audio_embeddings = torch.from_numpy(audio_embeddings)
                    text_embeddings = torch.from_numpy(text_embeddings)

                    all_innerprods.append(norm_and_convolve(audio_embeddings, text_embeddings))

>>>>>>> 0dc86209a5acc08195be69b9f3664c54f621c9c5

                # for seq_idx in range(0,seq_len_text):
                #     for layer_idx in range(0, num_layers_text): 
                #         layer_group = text_group[f"tuple_{seq_idx}"]
                #         
                #         # Extract and fix sequence dimension
                #         tensor = torch.tensor(layer_group[f"tensor_{layer_idx}"][:])  # (1, 1, 5120)
                #         # tensor = tensor[:, :1, :].mean(dim=0)  # Take only first time step → (5120)

                #         text_embeddings[seq_idx, layer_idx , :] = tensor.numpy()  # Offset by 1

                # **Average Over Sequence Length (Handles Different seq_lens)**
                # audio_layerwise = np.mean(audio_embeddings[1:], axis=0)  # Shape: (num_layers-1, 5120)
                # text_layerwise = np.mean(text_embeddings[1:], axis=0)  # Shape: (num_layers-1, 5120)
                # input_audio_layerwise = np.mean(audio_embeddings[:1], axis=0)  # Shape: (num_layers-1, 5120)
                # input_text_layerwise = np.mean(text_embeddings[:1], axis=0)  # Shape: (num_layers-1, 5120)
                


                # Store the filtered data
                # filtered_data[UID] = {
                #     "text_similarity": text_similarity,
                #     "audio_embedding": audio_layerwise,  # Shape: (num_layers-1, 5120)
                #     "text_embedding": text_layerwise,  # Shape: (num_layers-1, 5120)
                #     "input_audio_embedding": input_audio_layerwise,  # Shape: (num_layers-1, 5120)
                #     "input_text_embedding": input_text_layerwise,  # Shape: (num_layers-1, 5120)
                # }
                filtered_data[UID] = {'innerprods' : all_innerprods}
                with open(args.output_path, "wb") as f:
<<<<<<< HEAD
                    pickle.dump(filtered_data, f)
=======
                        pickle.dump(filtered_data, f)
>>>>>>> 0dc86209a5acc08195be69b9f3664c54f621c9c5

            else:
                print(f"Skipping {UID} (Similarity: {text_similarity:.2f})")
                skip_count +=1

    # Save Filtered Data
    print(f"total number of skipped is {skip_count} from total {count}")

    #with open(args.output_path, "wb") as f:
    #    pickle.dump(filtered_data, f)

    print("Filtering complete. Saved filtered dataset.")

# T5 Chatbot Using PyTorch & Transformers
This project is a conversational AI chatbot built using the `t5-small` model from Hugging Face Transformers and trained with PyTorch. It takes user input and responds conversationally based on a fine-tuned model trained on a custom dialogue dataset.

## Features
- Fine-tunes a T5 model (`t5-small`) on conversational data.
- Handles custom [sos], [eos] token-style training or T5-style chat: prompt.
- Interactive chat loop with real-time user input.
- TPU, GPU, or CPU compatible.
- Simple dataset support using .txt files.

## Dataset Format
The dataset consists of two text files:
- input_texts.txt — Contains lines of user messages.
- label_texts.txt — Contains corresponding chatbot replies.

Each line in both files should be aligned, i.e., line i in input_texts.txt corresponds to line i in label_texts.txt.

Example:
input_texts.txt
[sos] Hello! [eos]
[sos] How are you? [eos]

label_texts.txt
[sos] Hi there! [eos]
[sos] I'm doing great, thanks! [eos]

## How It Works
1. Loads and cleans the data.
2. Adds optional T5-style chat: prefix.
3. Tokenizes the inputs and outputs.
4. Fine-tunes a pre-trained T5 model (`t5-small`) for text generation.
5. Saves the model and allows interactive chatting.

## Usage
Training
Inside the notebook or script:
python train.py  # if modularized
or use the notebook interactively

## Chat with the model
After training, you can interact via the terminal:
Chatbot is ready! Type 'exit' to quit.
You: Hello
Bot: Hi there! How can I help you?

## Requirements
- transformers
- torch
- torch_xla (for TPU support, optional)
- scikit-learn
- pandas (optional, for analysis)

## Install via:
pip install transformers torch scikit-learn

For TPU:
pip install torch_xla -f https://storage.googleapis.com/libtorchxla-releases/wheels/tpuvm/torch_xla-2.1-cp310-cp310-linux_x86_64.whl

Notes
- To use [sos]/[eos] style data, make sure the inference matches that format.
- Alternatively, you can replace those with T5-friendly chat: prompts.
- You can further improve performance with more data and a larger model (e.g., t5-base).

## Directory Structure
.
├── input_texts.txt
├── label_texts.txt
├── chatbot_t5.pth          # saved model
├── chatbot_notebook.ipynb  # training + inference notebook
└── README.docx

## Author
Built by [Your Name] — feel free to fork and extend.
Future Improvements
- Add Gradio UI for web chat
- Context-aware multi-turn dialogue
- Save conversation history
- Evaluate responses using BLEU/ROUGE


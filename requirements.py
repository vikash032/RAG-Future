# Core Libraries
torch==2.2.2
torchvision
torchaudio
streamlit
chromadb
requests
pandas
regex
arxiv
datasets
langchain
unstructured[local-inference]
pdfminer.six
pi-heif
pdf2image
pillow-heif
python-magic
Duckdb
Spacy
langgraph
dlt
opencv-python<4.12  # Avoid 4.12+ due to compatibility issues

numpy==1.26.4               # Compatible with torch & transformers
transformers==4.44.2        # Pinned for compatibility with sentence-transformers
sentence-transformers==2.2.2
setfit==0.7.0               # Works with sentence-transformers 2.2.2
onnxruntime

# HuggingFace Ecosystem
optimum
huggingface_hub==0.23.3

FlagEmbedding @ git+https://github.com/FlagOpen/FlagEmbedding.git

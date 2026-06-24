# ===== BLOCK 1: imports =====
import json                                            # read/write the chunks file (JSON format)
import faiss                                           # vector index for fast similarity search
from sentence_transformers import SentenceTransformer  # turns text into embedding vectors
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # loads + runs the local LLM
import torch                                           # backend the LLM runs on (handles GPU)

# ===== BLOCK 2: loading (top level — so all functions can see these) =====
with open("chunks.json", "r", encoding="utf-8") as f:  # open the saved chunks file
    all_chunks = json.load(f)                          # parse JSON into a Python list

index = faiss.read_index("faiss_index.bin")            # load the prebuilt vector index from disk
model = SentenceTransformer("all-MiniLM-L6-v2")        # embedding model — MUST match the index build
print("Loaded", len(all_chunks), "chunks and an index with", index.ntotal, "vectors")

model_name = "Qwen/Qwen2.5-3B-Instruct"                # the local instruction-tuned LLM
tokenizer = AutoTokenizer.from_pretrained(model_name)  # converts text <-> tokens

quant_config = BitsAndBytesConfig(                     # 4-bit loading settings
    load_in_4bit=True,                                 # compress weights to 4-bit (~6GB -> ~2.5GB)
    bnb_4bit_compute_dtype=torch.float16,              # do the math in 16-bit for quality
    bnb_4bit_quant_type="nf4"                          # nf4 = quantization format tuned for LLMs
)

llm = AutoModelForCausalLM.from_pretrained(            # load the language model (4-bit)
    model_name,
    quantization_config=quant_config,                  # apply the 4-bit config
    device_map="auto"                                  # place it fully on the GPU
)
print("Model loaded on:", llm.device)                  # confirm GPU placement            # confirm it landed on the GPU

# ===== BLOCK 3: functions =====
def retrieve(question, k=5):                            # find the k most relevant chunks
    question_embedding = model.encode([question]).astype("float32")  # embed question (float32 for FAISS)
    distances, indices = index.search(question_embedding, k)  # search: distances + row positions

    results = []                                       # collect matching chunk records
    for i in indices[0]:                               # indices[0] = row numbers for our one question
        results.append(all_chunks[i])                  # map each row number back to its chunk
    return results                                     # return the k chunks

def build_prompt(question, chunks):                    # assemble the grounded prompt
    context = ""                                       # will hold retrieved chunks as text
    for c in chunks:                                   # loop over each retrieved chunk
        context += f"[Document {c['agora_id']}]\n{c['text']}\n\n"  # tag each with source doc

    prompt = f"""You are a helpful assistant answering questions about AI governance documents.
Use ONLY the context below to answer the question. 
If the answer is not in the context, say "I could not find this in the provided documents."
Cite the document numbers you used.

Context:
{context}
Question: {question}

Answer:"""                                              # enforces grounding + anti-hallucination + citations
    return prompt

def generate_answer(question, chunks):                 # prompt -> LLM -> answer
    prompt = build_prompt(question, chunks)            # build the grounded prompt

    messages = [{"role": "user", "content": prompt}]   # wrap in chat format the model expects
    text = tokenizer.apply_chat_template(              # apply Qwen's chat template
        messages,
        tokenize=False,                                # return text, not yet tokenized
        add_generation_prompt=True                     # cue the model to answer
    )

    inputs = tokenizer(text, return_tensors="pt").to(llm.device)  # tokenize + move to GPU

    outputs = llm.generate(                            # the actual generation
        **inputs,
        max_new_tokens=400,                            # cap answer length
        do_sample=False                                # greedy = deterministic, factual
    )

    response = tokenizer.decode(                       # tokens back to text
        outputs[0][inputs.input_ids.shape[1]:],        # slice off the prompt, keep only the answer
        skip_special_tokens=True                       # drop chat markers
    )
    return response

# ===== BLOCK 4: the CLI, wrapped in run_app() =====
def run_app():                                         # the interactive query loop
    print("\nAI Governance RAG — ask a question (type 'quit' to exit)\n")  # header
    while True:                                        # loop until user quits
        question = input("Question: ")                 # wait for input
        if question.lower() == "quit":                 # quit check (case-insensitive)
            break                                      # exit the loop
        chunks = retrieve(question)                    # 1. retrieve relevant chunks
        answer = generate_answer(question, chunks)     # 2. generate grounded answer
        print("\nAnswer:\n" + answer + "\n")           # 3. print it

# ===== run directly =====
if __name__ == "__main__":                             # only when run directly
    run_app()                                          # launch the CLI
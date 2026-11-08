from functools import lru_cache

import os

import threading

from huggingface_hub import hf_hub_download

import numpy as np

import onnxruntime as ort

from tokenizers import Tokenizer

from app.core.config import settings

from app.schemas.retriever import RetrievedChunks


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


class ONNXReranker:
    def __init__(self, model_id: str, filename: str):
        if os.path.isfile(filename):
            model_path = filename
    
        else:
            model_path = hf_hub_download(repo_id=model_id, filename=filename)

        opts = ort.SessionOptions()
    
        opts.intra_op_num_threads = 2
    
        opts.inter_op_num_threads = 1
    
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Safeguard 1: Disable CPU memory arena caching
        opts.enable_cpu_mem_arena = False

        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )

        self.tokenizer = Tokenizer.from_pretrained(model_id)
    
        self.tokenizer.enable_truncation(max_length=512)
    
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        self.input_names = [node.name for node in self.session.get_inputs()]
        
        # Safeguard 2: Thread lock to serialize forward passes under high concurrency
        self._lock = threading.Lock()

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []

        encodings = self.tokenizer.encode_batch(pairs)
    
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    
        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = np.array(
                [e.type_ids for e in encodings], dtype=np.int64
            )

        # Mutex-protected execution to prevent concurrent memory buffer duplication
        with self._lock:
            outputs = self.session.run(None, inputs)

        logits = outputs[0].flatten()
    
        scores = sigmoid(logits)
    
        return scores.tolist()


@lru_cache(maxsize=1)
def load_reranker() -> ONNXReranker:
    return ONNXReranker(
        model_id=settings.reranker_model,
        filename=settings.reranker_filename,
    )


def rerank_chunks(
    question: str,
    chunks: list[RetrievedChunks],
    top_k: int,
) -> list[RetrievedChunks]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if not chunks:
        return []

    reranker = load_reranker()

    question_chunk_pairs = [
        (question, str(chunk["text"]))
        for chunk in chunks
    ]

    rerank_scores = reranker.predict(question_chunk_pairs)

    ranked_chunks: list[RetrievedChunks] = []
    
    for chunk, score in zip(chunks, rerank_scores):
        ranked_chunk = chunk.copy()
        
        ranked_chunk["rerank_score"] = round(float(score), 4)
        
        ranked_chunks.append(ranked_chunk)

    ranked_chunks.sort(key=lambda chunk: float(chunk["rerank_score"]), reverse=True)

    return ranked_chunks[:top_k]
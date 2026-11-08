import os

import threading

from functools import lru_cache

from huggingface_hub import hf_hub_download

import numpy as np

import onnxruntime as ort

from tokenizers import Tokenizer

from app.core.config import settings


class ONNXEmbedder:
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

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        encodings = self.tokenizer.encode_batch(texts)
        
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

        token_embeddings = outputs[0]  # Shape: (batch_size, seq_len, 384)

        # Mean pooling over attention mask
        mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        
        embeddings = sum_embeddings / sum_mask

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        normalized = embeddings / np.clip(norms, a_min=1e-9, a_max=None)
        
        return normalized.tolist()


@lru_cache(maxsize=1)
def get_embedder() -> ONNXEmbedder:
    return ONNXEmbedder(
        model_id=settings.embedding_model,
        filename=settings.embedding_filename,
    )


def create_embeddings(text: list[str]) -> list[list[float]]:
    return get_embedder().encode(text)


def create_question_embeddings(question: str) -> list[float]:
    result = get_embedder().encode([question])
    return result[0] if result else []
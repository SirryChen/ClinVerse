"""Train and retrieve episodic memories from completed clinical encounters."""
from __future__ import annotations
from doctor_train.embeddings import EmbeddingEncoder
from doctor_train.memrl import (load_memory_bank, save_memory_bank, add_or_replace_memory,
                                retrieve_memories, intent_from_visible, format_memory_context,
                                update_memory_values, reward_from_scores)


class MemoryTrainer:
    def __init__(self, state_path=None, *, config, top_k=3, recall_k=20, alpha=0.1):
        self.encoder = EmbeddingEncoder(config['llm_providers']['embedding'])
        self.bank = load_memory_bank(state_path, model=self.encoder.model)
        self.top_k, self.recall_k, self.alpha = top_k, recall_k, alpha
        self.selected_ids = []

    def context(self, episode) -> str:
        selected = []
        if self.bank['items']:
            query = self.encoder.encode([intent_from_visible(episode.visible_info)])[0]
            selected = retrieve_memories(self.bank, query, top_k=self.top_k, recall_k=self.recall_k)
        self.selected_ids = [item['memory_id'] for item in selected]
        return format_memory_context(selected)

    def update(self, record: dict) -> None:
        update_memory_values(self.bank, self.selected_ids, reward_from_scores(record['scores']), alpha=self.alpha)
        vector = self.encoder.encode([intent_from_visible(record['transcript']['initial_visible_info'])])[0]
        add_or_replace_memory(self.bank, record, vector)

    def save(self, path) -> None:
        save_memory_bank(self.bank, path)

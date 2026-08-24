"""Training backends organized by post-training stage."""

TRAINING_STAGE_ORDER = ("cpt", "sft", "dpo", "grpo", "distill")

__all__ = ["TRAINING_STAGE_ORDER"]

"""Alpamayo R1 inference engine.

Loads the model once at startup and provides run_inference() to generate
Chain-of-Causation reasoning and trajectory predictions.
"""

import logging

import torch

from alpamayo_r1 import helper
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

from . import config

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Wraps AlpamayoR1 model for repeated inference."""

    def __init__(self):
        self.model: AlpamayoR1 | None = None
        self.processor = None

    def load(self):
        """Load model and processor. Call once at startup."""
        logger.info("Loading Alpamayo R1 model from %s ...", config.MODEL_ID)
        self.model = AlpamayoR1.from_pretrained(
            config.MODEL_ID, dtype=torch.bfloat16
        ).to("cuda")
        self.model.eval()
        self.processor = helper.get_processor(self.model.tokenizer)
        logger.info("Model loaded successfully.")

    def run_inference(
        self, frames: torch.Tensor, ego_data: dict[str, torch.Tensor]
    ) -> dict:
        """Run Alpamayo inference on frames + ego data.

        Args:
            frames: (N, 3, H, W) float32 tensor of camera frames.
            ego_data: Dict with ego_history_xyz (1,1,16,3) and
                      ego_history_rot (1,1,16,3,3) tensors.

        Returns:
            Dict with 'coc' (str), 'trajectory_summary' (str),
            'pred_xyz' (list of [x,y,z]).
        """
        # Build VLM messages from frames
        messages = helper.create_message(frames)

        # Tokenize
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )

        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": ego_data["ego_history_xyz"],
            "ego_history_rot": ego_data["ego_history_rot"],
        }
        model_inputs = helper.to_device(model_inputs, "cuda")

        # Run inference
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = self.model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=config.TOP_P,
                temperature=config.TEMPERATURE,
                num_traj_samples=config.NUM_TRAJ_SAMPLES,
                max_generation_length=config.MAX_GENERATION_LENGTH,
                return_extra=True,
            )

        # Extract results
        # pred_xyz: (1, 1, num_samples, 64, 3)
        # extra["cot"]: (1, 1, num_samples) numpy array of strings
        coc_text = str(extra["cot"][0, 0, 0])
        trajectory = pred_xyz[0, 0, 0].cpu().tolist()  # (64, 3)

        # Summarize trajectory (first/last waypoint, total distance)
        first = trajectory[0]
        last = trajectory[-1]
        total_dist = sum(
            sum((trajectory[i][d] - trajectory[i - 1][d]) ** 2 for d in range(3)) ** 0.5
            for i in range(1, len(trajectory))
        )
        summary = (
            f"Predicted {len(trajectory)} waypoints over 6.4s. "
            f"Start: ({first[0]:.1f}, {first[1]:.1f}, {first[2]:.1f})m, "
            f"End: ({last[0]:.1f}, {last[1]:.1f}, {last[2]:.1f})m, "
            f"Total distance: {total_dist:.1f}m"
        )

        return {
            "coc": coc_text,
            "trajectory_summary": summary,
            "pred_xyz": trajectory,
        }

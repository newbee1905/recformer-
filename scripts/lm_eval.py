import torch
import torch.nn.functional as F
from lm_eval.api.model import LM
from lm_eval import evaluator, tasks
from transformers import AutoTokenizer
from omegaconf import OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
import os

from model.recurrent_decoder import RecurrentDecoder


class RecurrentDecoderWrapper(LM):
	def __init__(self, checkpoint_path, config_path, device="cuda"):
		super().__init__()
		self.device = device

		# Load model configuration from the training run
		config = OmegaConf.load(config_path)

		# Adjust vocab_size to be divisible by 128, same as in training
		initial_vocab_size = AutoTokenizer.from_pretrained(config.model.tokenizer_path).vocab_size
		adjusted_vocab_size = (initial_vocab_size + 127) // 128 * 128
		config.model.vocab_size = adjusted_vocab_size

		# Load model architecture
		self.model = RecurrentDecoder(config.model)
		self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
		self.model.to(self.device)
		self.model.eval()

		self.tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_path)
		self.tokenizer.pad_token = self.tokenizer.eos_token

	@property
	def vocab_size(self):
		return self.tokenizer.vocab_size

	def loglikelihood(self, requests):
		res = []
		for context, continuation in requests:
			context_enc = self.tokenizer(context, return_tensors="pt")["input_ids"].to(self.device)
			continuation_enc = self.tokenizer(continuation, return_tensors="pt")["input_ids"].to(self.device)

			inp = torch.cat([context_enc, continuation_enc], dim=1)

			with torch.no_grad():
				logits, _ = self.model(inp)

			logits = logits[:, :-1, :]
			inp_target = inp[:, 1:]

			cont_len = continuation_enc.shape[1]
			continuation_logits = logits[:, -cont_len:, :]
			continuation_targets = inp_target[:, -cont_len:]

			log_probs = F.log_softmax(continuation_logits, dim=-1)

			token_log_probs = torch.gather(log_probs, 2, continuation_targets.unsqueeze(-1)).squeeze(-1)

			greedy_token_log_probs = token_log_probs.sum().item()

			res.append((greedy_token_log_probs, False))

		return res

	def generate_until(self, requests):
		pass

	def loglikelihood_rolling(self, requests):
		pass


@hydra.main(config_path="../config", config_name="train_config", version_base="1.3")
def main(cfg):
	"""
	Main function to run lm-eval-harness on a trained model.

	Usage:
	python scripts/lm_eval.py checkpoint_dir=outputs/YYYY-MM-DD/HH-MM-SS
	"""
	checkpoint_dir = cfg.checkpoint_dir
	checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
	config_path = os.path.join(checkpoint_dir, ".hydra", "config.yaml")

	if not os.path.exists(checkpoint_path) or not os.path.exists(config_path):
		raise FileNotFoundError(
			f"Checkpoint or config not found in {checkpoint_dir}. "
			"Please provide a valid checkpoint directory via 'checkpoint_dir=<path>'."
		)

	# Instantiate the model wrapper
	my_model = RecurrentDecoderWrapper(
		checkpoint_path=checkpoint_path,
		config_path=config_path,
	)

	# Initialize the task manager and select tasks
	task_manager = tasks.TaskManager()

	results = evaluator.simple_evaluate(model=my_model, tasks=["hellaswag"], num_fewshot=0, batch_size=8, device="cuda")

	print(evaluator.make_table(results))


if __name__ == "__main__":
	main()

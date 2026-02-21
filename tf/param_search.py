import argparse
import copy
import csv
import glob
import os
import subprocess
import uuid

import yaml

# Try to import TensorBoard for reading event files
try:
	from tensorboard.backend.event_processing import event_accumulator
	HAS_TENSORBOARD = True
except ImportError:
	HAS_TENSORBOARD = False
	print("Warning: tensorboard not installed, cannot extract metrics from logs")


def build_policy_value_ratio_variants(base_cfg):
	"""Yield (suffix, cfg_dict) pairs with different policy/value ratios.
	
	Goal: Find the best balance between policy and value learning.
	We keep the TOTAL weight sum constant to avoid implicit LR changes.
	
	Baseline sum: policy(1) + policy_soft(8) + 5*value(1) = 14
	We redistribute this budget differently.
	Refer to the paper for the values chosen as baseline.
	"""
	base_name = base_cfg["name"]
	
	# Total budget to distribute (keeps gradient scale similar)
	TOTAL_BUDGET = 14.0
	
	# Format: (policy_share, suffix)
	# policy_share = fraction of budget for policy heads
	# remaining goes to value heads
	ratios = [
		(0.64, "baseline"),        # 9/14 policy, 5/14 value (standard)
		(0.50, "balanced"),        # 50-50 split
		(0.35, "val_focus"),       # More value
		(0.25, "val_strong"),      # Even more value
		(0.75, "pol_focus"),       # More policy (for comparison)
	]
	
	for pol_share, suffix in ratios:
		# Make a deep copy of the base config to modify
		cfg = copy.deepcopy(base_cfg)

		# lw = loss_weights dict for convenience
		lw = cfg["training"]["loss_weights"]
		
		pol_budget = TOTAL_BUDGET * pol_share
		val_budget = TOTAL_BUDGET * (1 - pol_share)
		
		# Distribute policy budget (keep 1:8 ratio between policy and policy_soft)
		lw["policy"] = pol_budget / 9.0
		lw["policy_soft"] = pol_budget * 8.0 / 9.0
		
		# Distribute value budget equally across 5 value heads
		val_each = val_budget / 5.0
		lw["value_winner"] = val_each
		lw["value_q"] = val_each
		lw["value_st"] = val_each
		lw["value_q_err"] = val_each
		lw["value_st_err"] = val_each
		
		cfg["name"] = f"{base_name}-{suffix}"
		yield suffix, cfg


def build_value_head_variants(base_cfg):
	"""Yield (suffix, cfg_dict) pairs focusing on value head composition.
	
	Goal: Find which value targets are most useful with limited data.
	All variants sum to 5.0 (matching paper baseline) to keep gradient scale constant.
	"""
	base_name = base_cfg["name"]
	
	# All variants sum to 5.0 for fair comparison
	variants = [
		# Baseline: equal weights (5 × 1.0 = 5.0)
		{"value_winner": 1.0, "value_q": 1.0, "value_st": 1.0, 
		 "value_q_err": 1.0, "value_st_err": 1.0, "suffix": "val_baseline"},
		
		# Focus on WDL (winner) - most direct signal (2.0 + 4×0.75 = 5.0)
		{"value_winner": 2.0, "value_q": 0.75, "value_st": 0.75,
		 "value_q_err": 0.75, "value_st_err": 0.75, "suffix": "val_wdl_focus"},
		
		# Focus on Q/ST values (0.5 + 2×1.5 + 2×0.75 = 5.0)
		{"value_winner": 0.5, "value_q": 1.5, "value_st": 1.5,
		 "value_q_err": 0.75, "value_st_err": 0.75, "suffix": "val_q_focus"},
		
		# Disable error heads, boost others (5/3 ≈ 1.67 each, sum = 5.0)
		{"value_winner": 1.67, "value_q": 1.67, "value_st": 1.66,
		 "value_q_err": 0.0, "value_st_err": 0.0, "suffix": "val_no_err"},
		
		# Strong error focus (3×0.67 + 2×1.5 = 5.0)
		{"value_winner": 0.67, "value_q": 0.67, "value_st": 0.66,
		 "value_q_err": 1.5, "value_st_err": 1.5, "suffix": "val_err_focus"},
	]
	
	for v in variants:
		cfg = copy.deepcopy(base_cfg)
		lw = cfg["training"]["loss_weights"]
		
		lw["value_winner"] = v["value_winner"]
		lw["value_q"] = v["value_q"]
		lw["value_st"] = v["value_st"]
		lw["value_q_err"] = v["value_q_err"]
		lw["value_st_err"] = v["value_st_err"]
		
		suffix = v["suffix"]
		cfg["name"] = f"{base_name}-{suffix}"
		yield suffix, cfg


def extract_final_metrics(log_dir):
	"""Extract final metrics from TensorBoard event files.
	
	Returns dict with policy_accuracy, value_accuracy, etc.
	"""
	if not HAS_TENSORBOARD:
		return {}
	
	# Find event files in the test log directory
	event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
	if not event_files:
		print(f"No event files found in {log_dir}")
		return {}
	
	# Use the most recent event file
	event_file = sorted(event_files)[-1]
	
	try:
		ea = event_accumulator.EventAccumulator(event_file)
		ea.Reload()
		
		metrics = {}
		
		# Extract scalar metrics we care about
		tags_of_interest = [
			"Policy Accuracy",
			"Value Accuracy", 
			"Policy Loss",
			"Value Loss",
			"MSE Loss",
			"Total Loss",
			"ML Loss",
			"Soft Policy Loss"
		]
		
		available_tags = ea.Tags().get("scalars", [])
		
		for tag in tags_of_interest:
			if tag in available_tags:
				events = ea.Scalars(tag)
				if events:
					# Get the last value
					metrics[tag] = events[-1].value
		
		return metrics
		
	except Exception as e:
		print(f"Error reading event file: {e}")
		return {}


def run_variant(cfg, out_dir, train_script):
	"""Write a temporary YAML for this cfg and run train.py on it.
	
	Returns dict with config info and final metrics.
	"""

	os.makedirs(out_dir, exist_ok=True)

	# Derive a filename from cfg name plus a random tag.
	uid = uuid.uuid4().hex[:8]
	safe_name = cfg["name"].replace("/", "_")
	yaml_path = os.path.join(out_dir, f"{safe_name}-{uid}.yaml")

	with open(yaml_path, "w") as f:
		yaml.safe_dump(cfg, f, default_flow_style=False)

	cmd = [
		"python3",
		train_script,
		"--cfg",
		yaml_path
	]

	print("\n=== Running variant ===")
	print("YAML:", yaml_path)
	print("Command:", " ".join(cmd))

	subprocess.run(cmd, check=True)
	
	# Extract metrics from test logs
	# The log directory is typically based on the config name
	test_log_dir = os.path.join("leelalogs", f"{safe_name}-test")
	metrics = extract_final_metrics(test_log_dir)
	
	# Build result dict
	result = {
		"name": cfg["name"],
		"yaml_path": yaml_path,
		"policy": cfg["training"]["loss_weights"].get("policy", 1.0),
		"policy_soft": cfg["training"]["loss_weights"].get("policy_soft", 8.0),
		"value_winner": cfg["training"]["loss_weights"].get("value_winner", 1.0),
		"value_q": cfg["training"]["loss_weights"].get("value_q", 1.0),
		"value_st": cfg["training"]["loss_weights"].get("value_st", 1.0),
		"value_q_err": cfg["training"]["loss_weights"].get("value_q_err", 1.0),
		"value_st_err": cfg["training"]["loss_weights"].get("value_st_err", 1.0)
	}
	result.update(metrics)
	
	return result


def save_results_csv(results, csv_path):
	"""Save results list to CSV file."""
	if not results:
		return
	
	# Get all keys from results
	fieldnames = list(results[0].keys())
	
	with open(csv_path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(results)
	
	print(f"\nResults saved to: {csv_path}")


def find_best_config(results, metric="Value Accuracy"):
	"""Find the config with the best metric value.
	
	Returns the result dict of the best run, or None if no results.
	"""
	if not results:
		return None
	
	# Filter results that have the metric
	valid_results = [r for r in results if metric in r and r[metric] != "N/A"]
	if not valid_results:
		return None
	
	# Find best (highest value)
	best = max(valid_results, key=lambda r: r[metric])
	return best


def config_from_result(base_cfg, result):
	"""Reconstruct a config from a result dict.
	
	Takes the loss_weights from the result and applies them to base_cfg.
	"""
	cfg = copy.deepcopy(base_cfg)
	lw = cfg["training"]["loss_weights"]
	
	# Copy over the loss weights that were stored in result
	for key in ["policy", "policy_soft", "value_winner", "value_q", 
	            "value_st", "value_q_err", "value_st_err"]:
		if key in result:
			lw[key] = result[key]
	
	return cfg


def main():
	parser = argparse.ArgumentParser(
		description="Sequential parameter search: ratio -> value_heads (best from previous phase)",
	)
	parser.add_argument(
		"--base-cfg",
		type=str,
		required=True,
		help="Path to base YAML config (e.g. param_search/6m_search_base.yaml)",
	)
	parser.add_argument(
		"--train-script",
		type=str,
		required=True,
		help="Path to train.py (default: train.py in this directory)",
	)
	parser.add_argument(
		"--out-dir",
		type=str,
		default="paramsearch_runs",
		help="Directory to store generated YAMLs, logs.",
	)
	parser.add_argument(
		"--phase",
		type=str,
		choices=["ratio", "value_heads", "all"],
		default="all",
		help="Which search phase: ratio (policy vs value), value_heads (which value targets), or all",
	)
	parser.add_argument(
		"--optimize-for",
		type=str,
		default="Value Accuracy",
		help="Metric to optimize for when selecting best config (default: 'Value Accuracy')",
	)

	args = parser.parse_args()

	with open(args.base_cfg, "r") as f:
		base_cfg = yaml.safe_load(f)

	print("Base config name:", base_cfg.get("name"))
	print("Search phase:", args.phase)
	print("Optimizing for:", args.optimize_for)

	all_results = []
	current_base_cfg = base_cfg  # Start with original base config
	
	if args.phase == "all":
		# Phase 1: ratio search
		print(f"\n{'='*50}")
		print("=== PHASE 1: RATIO (policy vs value) ===")
		print(f"{'='*50}")
		
		phase1_results = []
		for suffix, cfg in build_policy_value_ratio_variants(current_base_cfg):
			try:
				result = run_variant(cfg, args.out_dir, args.train_script)
				phase1_results.append(result)
				all_results.append(result)
				
				csv_path = os.path.join(args.out_dir, "results.csv")
				save_results_csv(all_results, csv_path)
				
			except subprocess.CalledProcessError as e:
				print(f"Variant {cfg['name']} failed with return code {e.returncode}")
		
		# Find best from phase 1 and use as base for phase 2
		best_phase1 = find_best_config(phase1_results, args.optimize_for)
		if best_phase1:
			print(f"\n>>> Best from Phase 1: {best_phase1['name']}")
			print(f"    {args.optimize_for}: {best_phase1.get(args.optimize_for, 'N/A')}")
			current_base_cfg = config_from_result(base_cfg, best_phase1)
		else:
			print("\n>>> No valid results from Phase 1, using original base config")
		
		# Phase 2: value_heads search (using best from phase 1)
		print(f"\n{'='*50}")
		print("=== PHASE 2: VALUE_HEADS (using best ratio) ===")
		print(f"{'='*50}")
		
		for suffix, cfg in build_value_head_variants(current_base_cfg):
			try:
				result = run_variant(cfg, args.out_dir, args.train_script)
				all_results.append(result)
				
				csv_path = os.path.join(args.out_dir, "results.csv")
				save_results_csv(all_results, csv_path)
				
			except subprocess.CalledProcessError as e:
				print(f"Variant {cfg['name']} failed with return code {e.returncode}")
	
	else:
		# Single phase
		phases = {
			"ratio": build_policy_value_ratio_variants,
			"value_heads": build_value_head_variants,
		}
		
		print(f"\n{'='*50}")
		print(f"=== PHASE: {args.phase.upper()} ===")
		print(f"{'='*50}")
		
		builder = phases[args.phase]
		for suffix, cfg in builder(current_base_cfg):
			try:
				result = run_variant(cfg, args.out_dir, args.train_script)
				all_results.append(result)
				
				csv_path = os.path.join(args.out_dir, "results.csv")
				save_results_csv(all_results, csv_path)
				
			except subprocess.CalledProcessError as e:
				print(f"Variant {cfg['name']} failed with return code {e.returncode}")
	
	# Final summary
	print("\n" + "="*60)
	print("=== FINAL RESULTS ===")
	print("="*60)
	for r in all_results:
		pol_acc = r.get("Policy Accuracy", "N/A")
		val_acc = r.get("Value Accuracy", "N/A")
		print(f"{r['name']}: Policy={pol_acc}, Value={val_acc}")
	
	# Show overall best
	best_overall = find_best_config(all_results, args.optimize_for)
	if best_overall:
		print(f"\n>>> BEST OVERALL: {best_overall['name']}")
		print(f"    {args.optimize_for}: {best_overall.get(args.optimize_for, 'N/A')}")


if __name__ == "__main__":
	main()
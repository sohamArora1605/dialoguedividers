import os
import json
import torch
from pathlib import Path
from pyannote.audio import Pipeline

def run_sad(input_dir, output_dir):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Using device:", device)
	if device.type == "cuda":
		print("GPU:", torch.cuda.get_device_name(0))

	pipeline = Pipeline.from_pretrained(
		"pyannote/voice-activity-detection"
	).to(device)

	pipeline.instantiate({
		"onset": 0.3,
		"offset": 0.3,
		"min_duration_on": 0.15,
		"min_duration_off": 0.3
	})

	os.makedirs(output_dir, exist_ok=True)

	wav_files = [
		f for f in os.listdir(input_dir)
		if f.lower().endswith(".wav")
	]

	if not wav_files:
		raise RuntimeError(f"No .wav files found in {input_dir}")

	for wav_file in wav_files:
		input_path = os.path.join(input_dir, wav_file)
		base = os.path.splitext(wav_file)[0]
		output_path = os.path.join(output_dir, base + ".json")

		print(f"Processing: {wav_file}")

		speech = pipeline(input_path)

		segments = [
			{"start": seg.start, "end": seg.end}
			for seg in speech.get_timeline()
		]

		with open(output_path, "w") as f:
			json.dump(segments, f, indent=2)

	print("SAD completed successfully.")

if __name__ == "__main__":
	REPO_ROOT = Path(__file__).resolve().parents[2]

	input_dir = REPO_ROOT / "Track_1_SD_DevData_1" / "Hindi" / "data" / "wav"
	output_dir = REPO_ROOT / "outputs" / "sad"

	run_sad(str(input_dir), str(output_dir))
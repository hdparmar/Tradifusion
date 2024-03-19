from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="new_itt_spec_3c", split="train")
print(dataset)
dataset.push_to_hub("hdparmar/irish-traditional-tunes")
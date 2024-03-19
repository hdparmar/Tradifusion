from inference import TradifusionPipeline

pipeline = TradifusionPipeline.load_checkpoint("hdparmar/tradfusion-v2")

# Start and end prompts
start_prompt = "An Irish traditional tune"
end_prompt = "An Irish traditional tune with fiddle lead"


generated_image = pipeline.riffuse(start_prompt, end_prompt, num_inference_steps=50, alpha=0.5)

# Save
generated_image.save("output_image.png")


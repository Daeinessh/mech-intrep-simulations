
# Importing the transformer lens package
import transformer_lens

# Load the gpt 2 model which is pretrained into the hooked transformer
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")

inputText = "Dinesh is a building a time "

# Run the model and get logits and activations of the models as output
logits, activations = model.run_with_cache(inputText)


print("ACTIVATION CACHE", activations)
attention_pattern = activations["pattern", 0, "attn"]

print("ATTENTION PATTERN SHAPTE : ", attention_pattern.shape)

inputTextToken = model.to_str_tokens(inputText)
print("INPUT TEXT AS TOKEN : ", inputTextToken)
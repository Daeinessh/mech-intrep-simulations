
# Importing the transformer lens package
import transformer_lens

# Load the gpt 2 model which is pretrained into the hooked transformer
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")


# Prompt the model and get the list of predicted tokens by rank
output = transformer_lens.utils.test_prompt("Dinesh has completed the code, He is writting unit ", "test", model)
print(output)


# Run the model and get logits and activations of the models as output
logits, activations = model.run_with_cache("Dinesh has completed the code, He is writting unit ")

print("LOGITS :", logits)
print("ACTIVATIONS :", activations)

activationCacheList = []
for key, value in activations.items():
    activationCacheList.append(
        {
            "layerInfo" : str(key),
            "layerShape" : str(value.shape)
        }
    )
    print("INTREP LOG : ", "KEY: " + str(key), "VALUE: " + str(value.shape))
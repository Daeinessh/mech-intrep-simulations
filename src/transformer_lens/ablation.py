
# Importing the transformer lens package
import transformer_lens

# Load the gpt 2 model which is pretrained into the hooked transformer
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")


# Defining the input prompt
inputText = "Her name was Alex Hart. Tomorrow at lunch time Alex"
# Run the prompt through the model and getting the outputs as logits
outputLogits = model(inputText)



# The logit dimensions are: [batch, position, vocab]
outputTokenLogits = outputLogits[0, -1]

# Fetching the indexes of the the top 5 biggest logit values
topOutputTokenLogitsIndexes = sorted(
    range(len(list(outputTokenLogits))), key=lambda i: outputTokenLogits[i], reverse=True)[:5]

# Setting up the rank variable
outputWordRank = 1
for topOutputTokenLogitsIndex in list(topOutputTokenLogitsIndexes):
    # Decode the the actual work from the index value of the logit in the output
    outputWord = model.tokenizer.decode(topOutputTokenLogitsIndex)
    print(f"RANK : {outputWordRank} / NORMAL MODEL OUTPUT : ", outputWord)
    outputWordRank += 1



# Initializing variables which decide where ablation happens in the network
layerOfAblation = 0
headIndexOfAblation = 8

# Defining a function which actually sets the selects activations to zero when prompting
def ablateTheActivations(value, hook):
    print(f"Shape of the value tensor: {value.shape}")
    value[:, :, headIndexOfAblation, :] = 0.
    return value

# Running the model through same input text, but with hooks in place
ablatedLogits = model.run_with_hooks(
    # prompt contains this text as input "Her name was Alex Hart. Tomorrow at lunch time Alex"
    inputText, 
    fwd_hooks=[(
        transformer_lens.utils.get_act_name("v", layerOfAblation), 
        ablateTheActivations
        )]
    )


# The logit dimensions are: [batch, position, vocab]
ablatedTokenLogits = ablatedLogits[0, -1]
# Fetching the indexes of the the top 5 biggest logit values
topAblatedTokenLogitsIndexes = sorted(
    range(len(list(ablatedTokenLogits))), key=lambda i: ablatedTokenLogits[i], reverse=True)[:5]

# Setting up the rank variable
ablatedOutputWordRank = 1
print("INPUT PROMPT : ", inputText)
for topAblatedTokenLogitsIndex in list(topAblatedTokenLogitsIndexes):
    outputWord = model.tokenizer.decode(topAblatedTokenLogitsIndex)
    print(f"RANK : {ablatedOutputWordRank} / ABLATED MODEL OUTPUT : ", outputWord)
    ablatedOutputWordRank += 1

# Reseting the hooks is very important, as they are global variables and exists until you clear them
model.reset_hooks()


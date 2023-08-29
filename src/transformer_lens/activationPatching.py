import torch
import transformer_lens

# Load a model (eg GPT-2 Small)
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")

# I don't know what this does
# model.set_use_attn_result(True)


prompts = [
    'When John and Mary went to the shops, John gave the bag to', 
    'When John and Mary went to the shops, Mary gave the bag to', 

    'When Tom and James went to the park, James gave the ball to', 
    'When Tom and James went to the park, Tom gave the ball to', 

    'When Dan and Sid went to the shops, Sid gave an apple to', 
    'When Dan and Sid went to the shops, Dan gave an apple to', 

    'After Martin and Amy went to the park, Amy gave a drink to', 
    'After Martin and Amy went to the park, Martin gave a drink to'
]

answers = [
    (' Mary', ' John'), 
    (' John', ' Mary'), 
    (' Tom', ' James'), 
    (' James', ' Tom'), 
    (' Dan', ' Sid'),
    (' Sid', ' Dan'), 
    (' Martin', ' Amy'), 
    (' Amy', ' Martin')
]


cleanTokens = model.to_tokens(prompts)

# Swap each adjacent pair, with a hacky list comprehension
corrupted_tokens = cleanTokens[
    [(i+1 if i%2==0 else i-1) for i in range(len(cleanTokens)) ]
    ]

print("Clean string 0", model.to_string(cleanTokens[0]))
print("Corrupted string 0", model.to_string(corrupted_tokens[0]))

answer_token_indices = torch.tensor([[model.to_single_token(answers[i][j]) for j in range(2)] for i in range(len(answers))], device=model.cfg.device)
print("Answer token indices", answer_token_indices)

cleanLogits, cleanCache = model.run_with_cache(cleanTokens)


def get_logit_diff(logits, answer_token_indices=answer_token_indices):
    if len(logits.shape)==3:
        # Get final logits only
        logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()

clean_logits, clean_cache = model.run_with_cache(cleanTokens)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)



clean_logit_diff = get_logit_diff(clean_logits, answer_token_indices).item()
print(f"Clean logit diff: {clean_logit_diff:.4f}")

corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_token_indices).item()
print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")


CLEAN_BASELINE = clean_logit_diff
CORRUPTED_BASELINE = corrupted_logit_diff
def ioi_metric(logits, answer_token_indices=answer_token_indices):
    return (get_logit_diff(logits, answer_token_indices) - CORRUPTED_BASELINE) / (CLEAN_BASELINE  - CORRUPTED_BASELINE)

print(f"Clean Baseline is 1: {ioi_metric(clean_logits).item():.4f}")
print(f"Corrupted Baseline is 0: {ioi_metric(corrupted_logits).item():.4f}")



# The logit dimensions are: [batch, position, vocab]
outputTokenLogits = cleanLogits[0, -1]

# Fetching the indexes of the the top 5 biggest logit values
topOutputTokenLogitsIndexes = sorted(
    range(len(list(outputTokenLogits))), key=lambda i: outputTokenLogits[i], reverse=True)[:5]

# Setting up the rank variable
outputWordRank = 1
for topOutputTokenLogitsIndex in list(topOutputTokenLogitsIndexes):
    # Decode the the actual work from the index value of the logit in the output
    outputWord = model.tokenizer.decode(topOutputTokenLogitsIndex)
    print(f"RANK : {outputWordRank} / CLEAN MODEL OUTPUT : ", outputWord)
    outputWordRank += 1




corruptedLogits, corruptedCache = model.run_with_cache(corrupted_tokens)

# The logit dimensions are: [batch, position, vocab]
outputTokenLogits = corruptedLogits[0, -1]

# Fetching the indexes of the the top 5 biggest logit values
topOutputTokenLogitsIndexes = sorted(
    range(len(list(outputTokenLogits))), key=lambda i: outputTokenLogits[i], reverse=True)[:5]

# Setting up the rank variable
outputWordRank = 1
for topOutputTokenLogitsIndex in list(topOutputTokenLogitsIndexes):
    # Decode the the actual work from the index value of the logit in the output
    outputWord = model.tokenizer.decode(topOutputTokenLogitsIndex)
    print(f"RANK : {outputWordRank} / CORRUPTED MODEL OUTPUT : ", outputWord)
    outputWordRank += 1




# Activation patching the model

# Whether to do the runs by head and by position, which are much slower
DO_SLOW_RUNS = True

resid_pre_act_patch_results = transformer_lens.patching.get_act_patch_resid_pre(model, corrupted_tokens, cleanCache, ioi_metric)

print("", resid_pre_act_patch_results)
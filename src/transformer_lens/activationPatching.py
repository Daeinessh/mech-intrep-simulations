import torch
import transformer_lens

# Load a model (eg GPT-2 Small)
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")
model.set_use_attn_result(True)


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

clean_tokens = model.to_tokens(prompts)


# Swap each adjacent pair, with a hacky list comprehension
corrupted_tokens = clean_tokens[
    [(i+1 if i%2==0 else i-1) for i in range(len(clean_tokens)) ]
    ]

print("Clean string 0", model.to_string(clean_tokens[0]))
print("Corrupted string 0", model.to_string(corrupted_tokens[0]))

answer_token_indices = torch.tensor([[model.to_single_token(answers[i][j]) for j in range(2)] for i in range(len(answers))], device=model.cfg.device)
print("Answer token indices", answer_token_indices)

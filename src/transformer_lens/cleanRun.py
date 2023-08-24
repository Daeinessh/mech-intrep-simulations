import transformer_lens

# Load a model (eg GPT-2 Small)
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")


logits = model("Famous computer scientist Alan")

# The logit dimensions are: [batch, position, vocab]
next_token_logits = logits[0, -1]
next_token_prediction = next_token_logits.argmax()
next_word_prediction = model.tokenizer.decode(next_token_prediction)

print("MODEL OUTPUT :", next_word_prediction)


# Finding the rank of the text which given as second param for the prompt given at parameter one for the model given at 3rd param
transformer_lens.utils.test_prompt("Her name was Alex Hart. Tomorrow at lunch time Alex", " Hart", model)
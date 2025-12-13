from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

model_id = "google/paligemma-3b-pt-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

breakpoint()
prefix_output = model.language_model.forward(
    inputs_embeds=inputs_embeds[0],
    attention_mask=attention_mask,
    position_ids=position_ids,
    adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
)
prefix_past_key_values = prefix_output.past_key_values
# prefix_output to be used for the language head
# shape: [batch_size, seq_len, hidden_size] with hidden_size = 2048
prefix_output = prefix_output.last_hidden_state

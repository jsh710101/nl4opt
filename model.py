from torch.nn import Embedding
from transformers import BartForConditionalGeneration


class MyBartForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.token_type_embeddings = Embedding(10, config.d_model)
        self.token_type_embeddings.weight.data.fill_(0)

    def get_inputs_embeds(self, input_ids, token_type_ids):
        return self.model.encoder.embed_scale * self.model.shared(input_ids) + 5 * self.token_type_embeddings(token_type_ids)

    def forward(self, input_ids=None, token_type_ids=None, inputs_embeds=None, **kwargs):
        if (input_ids is not None) and (token_type_ids is not None) and (inputs_embeds is None):
            inputs_embeds = self.get_inputs_embeds(input_ids, token_type_ids)
        return super().forward(inputs_embeds=inputs_embeds, **kwargs)

    def generate(self, input_ids, token_type_ids, **kwargs):
        inputs_embeds = self.get_inputs_embeds(input_ids, token_type_ids)
        return super().generate(inputs_embeds=inputs_embeds, **kwargs)

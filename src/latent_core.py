import torch
from torch import nn
from torch.nn import functional as F

def insert_slots(inputs, prefix_length, slots, placeholder_id):
    result = dict(inputs)
    ids = inputs['input_ids']
    placeholder = ids.new_full((1, slots), placeholder_id)
    result['input_ids'] = torch.cat((ids[:, :prefix_length], placeholder, ids[:, prefix_length:]), 1)
    mask = inputs['attention_mask']
    result['attention_mask'] = torch.cat((mask[:, :prefix_length], mask.new_ones((1, slots)), mask[:, prefix_length:]), 1)
    return result

def pool_segments(states, slots):
    n = states.shape[0]
    assert n > 0
    result = []
    for i in range(slots):
        start = (i * n) // slots
        end = max(start + 1, ((i + 1) * n) // slots)
        result.append(states[start:end].float().mean(0))
    return torch.stack(result)

def alignment_loss(student, teacher):
    student = F.layer_norm(student.float(), (student.shape[-1],))
    teacher = F.layer_norm(teacher.detach().float(), (teacher.shape[-1],))
    cosine = (1 - F.cosine_similarity(student, teacher, dim=-1)).mean()
    return cosine + .5 * F.smooth_l1_loss(student, teacher)

class LatentPolicy(nn.Module):
    def __init__(self, model, slots=8, seed=42):
        super().__init__()
        self.model = model
        self.slots = slots
        embedding = model.get_input_embeddings()
        generator = torch.Generator(device='cpu').manual_seed(seed + 10000)
        initial = torch.randn(slots, embedding.weight.shape[1], generator=generator) * .02
        self.latent_embeddings = nn.Parameter(initial.to(embedding.weight.device))
        self.slot_start = None
        self.prefill_only = False
        self.hook = embedding.register_forward_hook(self._replace_slots)

    def _replace_slots(self, module, args, output):
        if self.slot_start is None:
            return output
        s = self.slot_start
        if self.prefill_only:
            self.slot_start = None
        assert output.shape[0] == 1 and output.shape[1] >= s + self.slots
        latent = self.latent_embeddings.to(output.dtype).unsqueeze(0)
        return torch.cat((output[:, :s], latent, output[:, s + self.slots:]), 1)

    def hidden(self, inputs, slot_start):
        assert self.slot_start is None
        self.slot_start = slot_start
        try:
            return self.model.get_base_model().model(**inputs, use_cache=False, return_dict=True).last_hidden_state
        finally:
            self.slot_start = None

    def forward(self, inputs, labels, prefix_length, teacher_target=None, alignment_weight=0.):
        h = self.hidden(inputs, prefix_length)
        answer_start = prefix_length + self.slots
        logits = self.model.get_output_embeddings()(h[0, answer_start - 1:-1])
        assert logits.shape[0] == labels.numel()
        core = F.cross_entropy(logits.float(), labels)
        aux = alignment_loss(h[0, prefix_length:answer_start], teacher_target) if teacher_target is not None else core.new_zeros(())
        return core + alignment_weight * aux, core, aux

    @torch.no_grad()
    def generate(self, inputs, prefix_length, max_new_tokens=1024):
        # Input is the observable prompt followed by the same eight private slots.
        assert inputs['input_ids'].shape[1] == prefix_length + self.slots
        previous = self.training
        self.eval()
        self.slot_start = prefix_length
        self.prefill_only = True
        try:
            return self.model.get_base_model().generate(
                **inputs, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True)
        finally:
            self.slot_start = None
            self.prefill_only = False
            self.train(previous)

"""CPU training and browser export for the tiny causal linear-attention model.

Usage: uv run --with torch python model/train.py
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tokenizer import TOKENIZER_VERSION

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "model"
SEED = 314159
D_MODEL, D_FEATURE, LAYERS, MAX_LENGTH = 32, 16, 2, 48
VOCAB = ["<unk>", ".", ",", "the", "a", "an", "and", "to", "up", "toward", "under", "on", "in", "with", "because", "so", "but", "then", "before", "after", "it", "he", "she", "i", "red", "blue", "green", "yellow", "small", "big", "robot", "android", "dog", "cat", "fox", "astronaut", "alice", "bob", "maya", "liam", "noah", "emma", "olivia", "key", "ball", "book", "map", "coin", "letter", "camera", "box", "door", "table", "desk", "garden", "room", "moon", "telescope", "picked", "walked", "dropped", "rolled", "chased", "barked", "handed", "gave", "thanked", "placed", "opened", "carried", "found", "held", "moved", "saw", "looked", "read", "closed", "loudly", "carefully"]
WORD_ID = {word: index for index, word in enumerate(VOCAB)}
COLORS = ["red", "blue", "green", "yellow"]
SUBJECTS = ["robot", "android", "dog", "cat", "fox", "astronaut"]
OBJECTS = ["key", "ball", "book", "map", "coin", "letter", "camera", "box"]
FEMALE_NAMES = ["alice", "maya", "emma", "olivia"]
MALE_NAMES = ["bob", "liam", "noah"]
LOCATIONS = ["door", "table", "desk", "garden", "room", "moon"]


@dataclass
class Example:
    words: list[str]
    pronoun_index: int
    antecedent_index: int
    causal: bool = True


def make_example(rng: random.Random) -> Example:
    color, subject, obj = rng.choice(COLORS), rng.choice(SUBJECTS), rng.choice(OBJECTS)
    location = rng.choice(LOCATIONS)
    family = rng.randrange(17)
    if family == 0:
        return Example(["the", color, subject, "picked", "up", "the", rng.choice(COLORS), obj, ".", "it", "walked", "toward", "the", location, "."], 9, 2)
    if family == 1:
        return Example(["the", subject, "dropped", "the", obj, ".", "it", "rolled", "under", "the", location, "."], 6, 4)
    if family == 2:
        return Example(["the", subject, "chased", "the", obj, ".", "it", "barked", "loudly", "."], 6, 1)
    if family == 3:
        return Example(["i", "saw", "an", "astronaut", "in", "the", "moon", "with", "a", "telescope", ".", "he", "walked", "toward", "the", location, "."], 11, 3)
    if family == 4:
        connector = rng.choice(["because", "so", "but", "then", "before", "after"])
        return Example(["the", subject, "chased", "the", obj, connector, "it", "barked", "loudly", "."], 6, 1)
    if family == 5:
        name, other = rng.choice(FEMALE_NAMES), rng.choice(MALE_NAMES)
        return Example([name, "handed", "the", obj, "to", other, ".", "he", "thanked", name, "."], 7, 5)
    if family == 6:
        name, other = rng.choice(FEMALE_NAMES), rng.choice(MALE_NAMES)
        return Example([name, "handed", other, "a", obj, ".", "she", "carried", "the", obj, "."], 6, 0)
    if family == 7:
        name, other = rng.choice(FEMALE_NAMES), rng.choice(MALE_NAMES)
        return Example([name, "gave", other, "a", obj, ".", other, "opened", "it", "carefully", "."], 8, 4)
    if family == 8:
        name = rng.choice(FEMALE_NAMES + MALE_NAMES)
        pronoun = "she" if name in FEMALE_NAMES else "he"
        return Example([name, "found", "a", obj, "in", "the", location, "then", pronoun, "held", "it", "."], 8, 0)
    people = FEMALE_NAMES + MALE_NAMES
    giver = rng.choice(people)
    recipient = rng.choice([person for person in people if person != giver])
    recipient_pronoun = "she" if recipient in FEMALE_NAMES else "he"
    if family == 9:
        return Example([giver, "handed", "a", obj, "to", recipient, ".", recipient_pronoun, "thanked", giver, "."], 7, 5)
    if family == 10:
        return Example([giver, "handed", recipient, "a", obj, ".", recipient_pronoun, "thanked", giver, "."], 6, 2)
    if family == 11:
        return Example([giver, "handed", "a", obj, "to", recipient, "and", recipient_pronoun, "thanked", giver, "."], 7, 5)
    if family == 12:
        return Example([giver, "gave", recipient, "a", obj, ".", recipient, "opened", "it", "carefully", "."], 8, 4)
    if family == 13:
        return Example([giver, "gave", "a", obj, "to", recipient, "and", recipient, "held", "it", "."], 9, 3)
    if family == 14:
        return Example([giver, "handed", "a", obj, "to", recipient, "because", recipient_pronoun, "held", "it", "."], 7, 5)
    # These are deliberately role-varied cataphora, not a memorized single sentence:
    # resolution is scored after the entire sentence, while the causal head is skipped.
    future_name = rng.choice(people)
    future_pronoun = "she" if future_name in FEMALE_NAMES else "he"
    other = rng.choice([person for person in people if person != future_name])
    if family == 15:
        return Example(["before", future_pronoun, "walked", ",", future_name, "thanked", other, "."], 1, 4, False)
    return Example(["after", future_pronoun, "looked", ",", future_name, "carried", "a", obj, "."], 1, 4, False)


def dataset(count: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    return [make_example(rng) for _ in range(count)]


class LinearAttentionLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(D_MODEL, D_FEATURE)
        self.k = nn.Linear(D_MODEL, D_FEATURE)
        self.v = nn.Linear(D_MODEL, D_MODEL)
        self.o = nn.Linear(D_MODEL, D_MODEL, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        state_s = x.new_zeros(batch, D_FEATURE, D_MODEL)
        state_z = x.new_zeros(batch, D_FEATURE)
        outputs = []
        for position in range(length):
            current = x[:, position]
            q_phi = F.elu(self.q(current)) + 1.000001
            k_phi = F.elu(self.k(current)) + 1.000001
            value = self.v(current)
            numerator = torch.bmm(q_phi.unsqueeze(1), state_s).squeeze(1)
            denominator = (q_phi * state_z).sum(-1, keepdim=True) + 1e-6
            context = numerator / denominator
            state_s = state_s + k_phi.unsqueeze(2) * value.unsqueeze(1)
            state_z = state_z + k_phi
            outputs.append(current + self.o(context))
        return torch.stack(outputs, 1)


class TinyCoref(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(len(VOCAB), D_MODEL)
        self.layers = nn.ModuleList(LinearAttentionLayer() for _ in range(LAYERS))
        self.reverse_layers = nn.ModuleList(LinearAttentionLayer() for _ in range(LAYERS))
        self.head = nn.Linear(D_MODEL, len(VOCAB), bias=False)  # retained for export compatibility; pointer is used for coreference
        self.pointer = nn.Parameter(torch.empty(D_MODEL, D_MODEL))
        self.pointer_full = nn.Parameter(torch.empty(D_MODEL, D_MODEL))
        nn.init.xavier_uniform_(self.pointer)
        nn.init.xavier_uniform_(self.pointer_full)

    def encode(self, ids: Tensor, layers: nn.ModuleList, reverse: bool = False) -> Tensor:
        if not reverse:
            hidden = self.embedding(ids)
            for layer in layers:
                hidden = layer(hidden)
            return hidden
        # Do not let right-padding become fake future context.  Process each
        # observed sequence length as one vectorized group, then restore order.
        output = self.embedding(ids).new_zeros(ids.size(0), ids.size(1), D_MODEL)
        lengths = (ids != WORD_ID["<unk>"]).sum(1)
        for length in lengths.unique().tolist():
            selected = lengths == length
            reversed_ids = ids[selected, :length].flip(1)
            hidden = self.embedding(reversed_ids)
            for layer in layers:
                hidden = layer(hidden)
            output[selected, :length] = hidden.flip(1)
        return output

    def forward(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        return self.encode(ids, self.layers), self.encode(ids, self.reverse_layers, reverse=True)

    def pointer_scores(self, hidden: Tensor, ids: Tensor, positions: Tensor) -> Tensor:
        query = hidden[torch.arange(hidden.size(0)), positions] @ self.pointer
        scores = torch.einsum("bd,bld->bl", query, hidden)
        entity_ids = torch.tensor([WORD_ID[word] for word in SUBJECTS + OBJECTS + FEMALE_NAMES + MALE_NAMES], device=ids.device)
        valid = torch.isin(ids, entity_ids) & (torch.arange(ids.size(1), device=ids.device).unsqueeze(0) < positions.unsqueeze(1))
        return scores.masked_fill(~valid, -1e9)

    def full_scores(self, forward: Tensor, backward: Tensor, ids: Tensor, positions: Tensor) -> Tensor:
        hidden = forward + backward
        query = hidden[torch.arange(hidden.size(0)), positions] @ self.pointer_full
        scores = torch.einsum("bd,bld->bl", query, hidden)
        entity_ids = torch.tensor([WORD_ID[word] for word in SUBJECTS + OBJECTS + FEMALE_NAMES + MALE_NAMES], device=ids.device)
        valid = torch.isin(ids, entity_ids) & (torch.arange(ids.size(1), device=ids.device).unsqueeze(0) != positions.unsqueeze(1))
        return scores.masked_fill(~valid, -1e9)


def batchify(examples: list[Example]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    length = max(len(example.words) for example in examples)
    ids = torch.zeros(len(examples), length, dtype=torch.long)
    positions = torch.zeros(len(examples), dtype=torch.long)
    targets = torch.zeros(len(examples), dtype=torch.long)
    causal = torch.zeros(len(examples), dtype=torch.bool)
    for row, example in enumerate(examples):
        ids[row, :len(example.words)] = torch.tensor([WORD_ID.get(word, 0) for word in example.words])
        positions[row] = example.pronoun_index
        targets[row] = example.antecedent_index
        causal[row] = example.causal
    return ids, positions, targets, causal


@torch.no_grad()
def accuracy(model: TinyCoref, examples: list[Example]) -> float:
    model.eval()
    ids, positions, targets, _ = batchify(examples)
    forward, backward = model(ids)
    logits = model.full_scores(forward, backward, ids, positions)
    return (logits.argmax(-1) == targets).float().mean().item()


def json_value(value: Tensor) -> list:
    return value.detach().cpu().float().tolist()


def export(model: TinyCoref, train_accuracy: float, validation_accuracy: float, test_accuracy: float) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    def export_layers(source: nn.ModuleList) -> list[dict[str, list]]:
        layers = []
        for layer in source:
            # PyTorch uses out×in; browser uses x[in] × W[in][out].
            layers.append({"wq": json_value(layer.q.weight.T), "wk": json_value(layer.k.weight.T), "wv": json_value(layer.v.weight.T), "wo": json_value(layer.o.weight.T), "bq": json_value(layer.q.bias), "bk": json_value(layer.k.bias), "bv": json_value(layer.v.bias)})
        return layers
    weights = {"embeddings": json_value(model.embedding.weight), "layers": export_layers(model.layers), "reverseLayers": export_layers(model.reverse_layers), "head": json_value(model.head.weight.T), "pointer": json_value(model.pointer), "pointerFull": json_value(model.pointer_full)}
    digest = hashlib.sha256(json.dumps(weights, sort_keys=True).encode()).hexdigest()[:12]
    config = {"version": "full-sentence-pointer-1", "dModel": D_MODEL, "dFeature": D_FEATURE, "maxLength": MAX_LENGTH, "epsilon": 1e-6, "layers": LAYERS, "featureMap": "ELU(x) + 1 + eps", "tokenizer": TOKENIZER_VERSION, "entityTokenIds": [WORD_ID[word] for word in SUBJECTS + OBJECTS + FEMALE_NAMES + MALE_NAMES], "weightsChecksum": digest, "training": {"status": "trained", "device": "cpu", "seed": SEED, "task": "synthetic pronoun antecedent position prediction with a forward and reverse encoder"}}
    metrics = {"status": "trained", "trainExamples": 18000, "validationExamples": 2000, "testExamples": 2000, "templateFamilies": 17, "trainAccuracy": train_accuracy, "validationAccuracy": validation_accuracy, "testAccuracy": test_accuracy, "task": "full-sentence antecedent position pointer at pronoun positions", "scope": "Tiny model trained only on controlled short English templates. The full-sentence resolver can consider entity tokens on either side of a pronoun; the live trace remains the forward causal pass."}
    (OUT / "model-config.json").write_text(json.dumps(config, indent=2))
    (OUT / "vocab.json").write_text(json.dumps(VOCAB, indent=2))
    (OUT / "weights.json").write_text(json.dumps(weights, separators=(",", ":")))
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT / "examples.json").write_text(json.dumps(["The red robot picked up the blue key. It walked toward the door.", "The dog dropped the ball. It rolled under the table.", "The dog chased the ball. It barked loudly.", "Alice handed the map to Bob. He thanked Alice.", "Before she walked, Maya thanked Bob.", "I saw an astronaut in the moon with a telescope."], indent=2))


def main() -> None:
    torch.manual_seed(SEED)
    model = TinyCoref()
    # The domain has deliberately few compositional combinations; this is enough
    # for reliable held-out instances without making CPU training burdensome.
    train, validation, test = dataset(18000, SEED), dataset(2000, SEED + 1), dataset(2000, SEED + 2)
    ids, positions, targets, causal = batchify(train)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.012, weight_decay=1e-4)
    for epoch in range(40):
        model.train()
        forward, backward = model(ids)
        full_logits = model.full_scores(forward, backward, ids, positions)
        causal_logits = model.pointer_scores(forward, ids, positions)
        loss = F.cross_entropy(full_logits, targets) + F.cross_entropy(causal_logits[causal], targets[causal])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if epoch % 10 == 9:
            print(f"epoch {epoch + 1:02d} loss={loss.item():.4f} validation={accuracy(model, validation):.3f}")
    train_acc, val_acc, test_acc = accuracy(model, train), accuracy(model, validation), accuracy(model, test)
    print(f"final train={train_acc:.3f} validation={val_acc:.3f} test={test_acc:.3f}")
    canonical = Example("the red robot picked up the blue key . it walked toward the door .".split(), 9, 2)
    canonical_ids, canonical_position, _, _ = batchify([canonical])
    canonical_forward, canonical_backward = model(canonical_ids)
    canonical_top = model.full_scores(canonical_forward, canonical_backward, canonical_ids, canonical_position)[0].argmax().item()
    print(f"canonical prediction={canonical.words[canonical_top]}")
    if test_acc < .90:
        raise RuntimeError(f"Held-out accuracy {test_acc:.3f} is below the 0.90 requirement.")
    export(model, train_acc, val_acc, test_acc)


if __name__ == "__main__":
    main()

"""
test_evaluator_constrained.py — regression test cho 2 bug thật da xay ra
tren Kaggle (R0/R1 that bai lan 2, sau khi sua xong mount path):

1. constrained_crf phai o CUNG device voi model (bug that: RuntimeError
   "tensors on cuda:0 and cpu" vi Evaluator duoc khoi tao TRUOC khi
   Trainer.__init__ goi model.to(device), va constrained_crf luc do la
   BAN SAO doc lap KHONG tu dong theo model.to() sau nay).
2. constrained_crf PHAI phan anh trong so HIEN TAI cua model.crf (bug
   tiem an: neu cache 1 lan luc __init__, moi lan decode sau se dung
   trong so ngau nhien luc khoi tao, khong bao gio phan anh model dang
   hoc qua tung epoch).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torchcrf import CRF

from src.evaluator import Evaluator

LABEL2ID = {"O": 0, "B-PER": 1, "I-PER": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class _FakeModel(nn.Module):
    """Model gia lap toi thieu: chi can .crf (CRF that) de Evaluator xay
    constrained_crf tu do -- khong can BERT that."""
    def __init__(self):
        super().__init__()
        self.crf = CRF(len(LABEL2ID), batch_first=True)


def test_constrained_crf_matches_model_device_even_when_built_before_model_to_device():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("Can GPU de tai hien dung bug device that (CPU-only luon 'khop' gia).")
    model = _FakeModel()
    evaluator = Evaluator(model, ID2LABEL, torch.device("cuda"), "/tmp",
                           use_constrained_decode=True, label2id=LABEL2ID)
    # Mo phong dung thu tu bug that: Evaluator khoi tao TRUOC, model.to(device) SAU.
    model.to("cuda")
    crf = evaluator.constrained_crf
    assert crf.transitions.device.type == "cuda"
    assert crf.start_transitions.device.type == "cuda"
    assert crf.end_transitions.device.type == "cuda"


def test_constrained_crf_reflects_latest_model_weights_not_a_stale_init_snapshot():
    model = _FakeModel()
    evaluator = Evaluator(model, ID2LABEL, torch.device("cpu"), "/tmp",
                           use_constrained_decode=True, label2id=LABEL2ID)

    crf_before = evaluator.constrained_crf
    val_before = crf_before.transitions[LABEL2ID["B-PER"], LABEL2ID["I-PER"]].item()

    # mo phong 1 buoc optimizer.step() thay doi trong so CRF that cua model
    with torch.no_grad():
        model.crf.transitions[LABEL2ID["B-PER"], LABEL2ID["I-PER"]] += 999.0

    crf_after = evaluator.constrained_crf
    val_after = crf_after.transitions[LABEL2ID["B-PER"], LABEL2ID["I-PER"]].item()

    assert val_after != val_before, (
        "constrained_crf dang la SNAPSHOT cu, khong phan anh trong so moi cap nhat cua model.crf"
    )
    assert abs(val_after - (val_before + 999.0)) < 1e-3


def test_constrained_crf_none_when_use_constrained_decode_false():
    model = _FakeModel()
    evaluator = Evaluator(model, ID2LABEL, torch.device("cpu"), "/tmp")
    assert evaluator.constrained_crf is None

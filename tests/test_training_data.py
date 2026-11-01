import importlib.util
import unittest

from probeforge.training_data import encode_completion, CompletionCollator, sequence_logps


class FakeTokenizer:
    eos_token = "!"
    def apply_chat_template(self, messages, **kwargs):
        return "prompt:"
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


class EncodingTests(unittest.TestCase):
    def test_prompt_mask_and_eos(self):
        row = encode_completion(FakeTokenizer(), [], "ok", 20)
        self.assertEqual(row["labels"][:7], [-100]*7)
        self.assertEqual(row["labels"][-3:], [111, 107, 33])
        with self.assertRaises(ValueError):
            encode_completion(FakeTokenizer(), [], "ok", 4)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Training dependencies not installed")
    def test_logps_padding_invariance_and_gradient(self):
        import torch
        torch.manual_seed(3)
        logits = torch.randn(1, 5, 7, requires_grad=True)
        labels = torch.tensor([[-100, -100, 2, 3, 4]])
        value = sequence_logps(logits, labels)
        expected = torch.log_softmax(logits[0, 1:4], -1)[torch.arange(3), torch.tensor([2, 3, 4])].sum()
        self.assertTrue(torch.allclose(value[0], expected))
        padded_logits = torch.cat([logits, torch.randn(1, 2, 7)], dim=1)
        padded_labels = torch.cat([labels, torch.tensor([[-100, -100]])], dim=1)
        self.assertTrue(torch.allclose(sequence_logps(padded_logits, padded_labels), value))
        (-value.sum()).backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()

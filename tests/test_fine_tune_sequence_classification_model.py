# Use Relative Import from the ..src/transformers/trainwithaccelerate package
from ..src.transformers.trainwithaccelerate import Trainer, split_dataset


class TestFineTuneSequenceClassificationModel(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.dataset = MyDataset("data_path", self.tokenizer)
        (
            self.train_subset,
            self.eval_subset,
            self.val_subset,
            self.test_subset,
        ) = split_dataset(
            self.dataset,
            train_ratio=0.8,
            eval_ratio=0.1,
            val_ratio=0.05,
            test_ratio=0.05,
            seed=42,
        )
        self.batch_size = 16
        self.train_dataloader = DataLoader(
            self.train_subset, batch_size=self.batch_size, shuffle=True
        )
        self.eval_dataloader = DataLoader(
            self.eval_subset, batch_size=self.batch_size, shuffle=False
        )
        self.val_dataloader = DataLoader(
            self.val_subset, batch_size=self.batch_size, shuffle=False
        )
        self.test_dataloader = DataLoader(
            self.test_subset, batch_size=self.batch_size, shuffle=False
        )
        self.trainer = Trainer(
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader,
            val_dataloader=self.val_dataloader,
            test_dataloader=self.test_dataloader,
        )

    def test_split_dataset(self):
        train_subset, eval_subset, val_subset, test_subset = split_dataset(
            self.dataset,
            train_ratio=0.8,
            eval_ratio=0.1,
            val_ratio=0.05,
            test_ratio=0.05,
            seed=42,
        )
        self.assertEqual(len(train_subset), 80)
        self.assertEqual(len(eval_subset), 10)
        self.assertEqual(len(val_subset), 5)
        self.assertEqual(len(test_subset), 5)

    def test_prepare(self):
        self.trainer.prepare()
        self.assertIsNotNone(self.trainer.model)
        self.assertIsNotNone(self.trainer.optimizer)
        self.assertIsNotNone(self.trainer.lr_scheduler)
        self.assertIsNotNone(self.trainer.progress_bar)

    def test_train(self):
        self.trainer.prepare()
        self.trainer.train()
        self.assertIsNotNone(self.trainer.model)

    def test_evaluate(self):
        self.trainer.prepare()
        self.trainer.train()
        self.trainer.model.eval()
        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            for batch in self.val_dataloader:
                batch = {k: v.to(self.trainer.device) for k, v in batch.items()}
                outputs = self.trainer.model(**batch)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                labels = batch["labels"]
                total_correct += (predictions == labels).sum().item()
                total_samples += len(labels)
            accuracy = total_correct / total_samples
            self.assertGreaterEqual(accuracy, 0.0)
            self.assertLessEqual(accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()

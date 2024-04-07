import os
import random
import torch
from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    get_scheduler,
    AutoTokenizer,
)
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from dotenv import load_dotenv
import unittest

load_dotenv()


class Trainer:
    """
    A class for training a sequence classification model using the Hugging Face Transformers library.

    Args:
        checkpoint (str): The path or identifier of the pre-trained checkpoint to use.
        train_dataloader (DataLoader): The data loader for the training set.
        eval_dataloader (DataLoader): The data loader for the evaluation set.
        val_dataloader (DataLoader): The data loader for the validation set.
        test_dataloader (DataLoader): The data loader for the test set.
        num_epochs (int, optional): The number of epochs to train for. Defaults to 3.
        lr (float, optional): The learning rate to use for the optimizer. Defaults to 3e-5.
    """

    def __init__(
        self,
        checkpoint=None,
        train_dataloader=None,
        eval_dataloader=None,
        val_dataloader=None,
        test_dataloader=None,
        num_epochs=None,
        lr=None,
    ):
        """
        Initializes a new instance of the Trainer class.

        Args:
            checkpoint (str): The path or identifier of the pre-trained checkpoint to use.
            train_dataloader (DataLoader): The data loader for the training set.
            eval_dataloader (DataLoader): The data loader for the evaluation set.
            val_dataloader (DataLoader): The data loader for the validation set.
            test_dataloader (DataLoader): The data loader for the test set.
            num_epochs (int, optional): The number of epochs to train for. Defaults to 3.
            lr (float, optional): The learning rate to use for the optimizer. Defaults to 3e-5.
        """
        self.checkpoint = checkpoint or os.getenv(
            "CHECKPOINT", "distilbert-base-uncased"
        )
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.num_epochs = num_epochs or int(os.getenv("NUM_EPOCHS", 3))
        self.lr = lr or float(os.getenv("LR", 3e-5))
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.accelerator = Accelerator()
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.progress_bar = None

    def prepare(self):
        """
        Initializes the model, optimizer, and learning rate scheduler.
        """
        if (
            self.train_dataloader is None
            or self.eval_dataloader is None
            or self.val_dataloader is None
            or self.test_dataloader is None
        ):
            raise ValueError("Data loaders not defined. Cannot prepare trainer.")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.checkpoint, num_labels=2
        )
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)
        (
            self.train_dataloader,
            self.eval_dataloader,
            self.val_dataloader,
            self.test_dataloader,
            self.model,
            self.optimizer,
        ) = self.accelerator.prepare(
            self.train_dataloader,
            self.eval_dataloader,
            self.val_dataloader,
            self.test_dataloader,
            self.model,
            self.optimizer,
        )
        num_training_steps = self.num_epochs * len(self.train_dataloader)
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        self.progress_bar = tqdm(range(num_training_steps))

    def train(self):
        """
        Trains the model for the specified number of epochs.

        Raises:
            ValueError: If the model, optimizer, learning rate scheduler, or progress bar is not initialized.
        """
        if (
            self.model is None
            or self.optimizer is None
            or self.lr_scheduler is None
            or self.progress_bar is None
        ):
            raise ValueError("Trainer not prepared. Call prepare() method first.")
        self.model.train()
        for epoch in range(self.num_epochs):
            for batch in self.train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                self.accelerator.backward(loss)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.progress_bar.update(1)


def split_dataset(
    dataset, train_ratio=0.8, eval_ratio=0.1, val_ratio=0.05, test_ratio=0.05, seed=42
):
    """
    Splits a dataset into training, evaluation, validation, and test subsets.

    Args:
        dataset (Dataset): The dataset to split.
        train_ratio (float, optional): The ratio of examples to use for training. Defaults to 0.8.
        eval_ratio (float, optional): The ratio of examples to use for evaluation. Defaults to 0.1.
        val_ratio (float, optional): The ratio of examples to use for validation. Defaults to 0.05.
        test_ratio (float, optional): The ratio of examples to use for testing. Defaults to 0.05.
        seed (int, optional): The random seed to use for shuffling the dataset. Defaults to 42.

    Returns:
        Tuple[Subset]: A tuple of four subsets for training, evaluation, validation, and test.
    """
    num_examples = len(dataset)
    indices = list(range(num_examples))
    random.seed(seed)
    random.shuffle(indices)
    train_size = int(train_ratio * num_examples)
    eval_size = int(eval_ratio * num_examples)
    val_size = int(val_ratio * num_examples)
    test_size = int(test_ratio * num_examples)
    train_indices = indices[:train_size]
    eval_indices = indices[train_size : train_size + eval_size]
    val_indices = indices[train_size + eval_size : train_size + eval_size + val_size]
    test_indices = indices[
        train_size
        + eval_size
        + val_size : train_size
        + eval_size
        + val_size
        + test_size
    ]
    train_subset = Subset(dataset, train_indices)
    eval_subset = Subset(dataset, eval_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    return train_subset, eval_subset, val_subset, test_subset


# Example usage
if __name__ == "__main__":
    from my_dataset import MyDataset

    # Load dataset
    data_path = os.getenv("DATA_PATH")
    tokenizer = AutoTokenizer.from_pretrained(
        os.getenv("TOKENIZER", "distilbert-base-uncased")
    )
    dataset = MyDataset(data_path, tokenizer)

    # Split dataset
    train_ratio = float(os.getenv("TRAIN_RATIO", 0.8))
    eval_ratio = float(os.getenv("EVAL_RATIO", 0.1))
    val_ratio = float(os.getenv("VAL_RATIO", 0.05))
    test_ratio = float(os.getenv("TEST_RATIO", 0.05))
    seed = int(os.getenv("SEED", 42))
    train_subset, eval_subset, val_subset, test_subset = split_dataset(
        dataset, train_ratio, eval_ratio, val_ratio, test_ratio, seed
    )

    # Create data loaders
    batch_size = int(os.getenv("BATCH_SIZE", 16))
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_subset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    # Create trainer
    trainer = Trainer(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
    )

    # Prepare trainer
    trainer.prepare()

    # Train model
    trainer.train()

    # Evaluate model on validation set
    trainer.model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for batch in val_dataloader:
            batch = {k: v.to(trainer.device) for k, v in batch.items()}
            outputs = trainer.model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            labels = batch["labels"]
            total_correct += (predictions == labels).sum().item()
            total_samples += len(labels)
        accuracy = total_correct / total_samples
        print(f"Validation accuracy: {accuracy:.4f}")

    # Evaluate model on test set
    trainer.model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for batch in test_dataloader:
            batch = {k: v.to(trainer.device) for k, v in batch.items()}
            outputs = trainer.model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            labels = batch["labels"]
            total_correct += (predictions == labels).sum().item()
            total_samples += len(labels)
        accuracy = total_correct / total_samples
        print(f"Test accuracy: {accuracy:.4f}")
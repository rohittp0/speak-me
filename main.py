import os
from pathlib import Path

import torch

from dataset import UnicodeData
from mingpt.model import GPT
from mingpt.trainer import Trainer

model_save_folder = "models"


def get_batch_end_callback(model: GPT, train_dataset: UnicodeData):
    # iteration callback
    def batch_end_callback(trainer: Trainer):

        if trainer.iter_num % 10 == 0:
            print(
                f"iter_dt {trainer.iter_dt * 1000:.2f}ms; "
                f"iter {trainer.iter_num}: train loss {trainer.loss.item()} "
                f"lr {trainer.scheduler.get_last_lr()[0]}",
                end=" " if trainer.iter_num % 100 == 0 else "\n",
            )

        if trainer.iter_num % 100 == 0:
            test_loss = trainer.test()
            print(f" Test loss {test_loss:.5f}")

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            with torch.no_grad():
                # sample from the model...
                context = train_dataset.encode_sequence("അതിവേഗം ")
                x = torch.tensor(context, dtype=torch.long)[None, ...].to(trainer.device)
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]

                completion = train_dataset.decode_sequence(y.tolist())
                print("Completion: ")
                print(completion)
            # save the latest model
            print("saving model")

            ckpt_path = os.path.join(model_save_folder, f"{model.model_type}-model.pt")
            trainer.save_checkpoint(ckpt_path)
            # revert model to training mode
            model.train()

    return batch_end_callback


def main():
    model_config = GPT.get_default_config()
    trainer_config = Trainer.get_default_config()

    manglish = UnicodeData("data", 1024)

    train_size = int(0.9 * len(manglish))
    # Created using indices from 0 to train_size.
    train_dataset = torch.utils.data.Subset(manglish, range(train_size))

    # Created using indices from train_size to train_size + test_size.
    test_dataset = torch.utils.data.Subset(manglish, range(train_size, len(manglish)))

    model_config.model_type = 'gpt-mini'
    model_config.vocab_size = manglish.get_vocab_size()
    model_config.block_size = manglish.block_size  # OpenAI's model block_size (i.e. input context length)
    model = GPT(model_config)

    trainer_config.batch_size = 4
    trainer_config.max_iters = 100_000
    trainer_config.num_workers = 6
    trainer_config.lr_decay_steps = 100
    trainer_config.lr_decay_gamma = 0.8

    trainer = Trainer(trainer_config, model, train_dataset, test_dataset)
    trainer.set_callback('on_batch_end', get_batch_end_callback(model, manglish))

    saved = os.path.join(model_save_folder, f"{model.model_type}-model.pt")
    if os.path.exists(saved):
        trainer.load_checkpoint(saved)
    else:
        Path(model_save_folder).mkdir(parents=True, exist_ok=True)

    trainer.run()


if __name__ == "__main__":
    main()

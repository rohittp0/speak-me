import os

import torch

from dataset import UnicodeData
from mingpt.model import GPT
from mingpt.trainer import Trainer


def get_batch_end_callback(model: GPT, train_dataset: UnicodeData):
    # iteration callback
    def batch_end_callback(trainer: Trainer):

        if trainer.iter_num % 10 == 0:
            print(
                f"iter_dt {trainer.iter_dt * 1000:.2f}ms; "
                f"iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}"
            )

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # sample from the model...
                context = train_dataset.encode_sequence("<EXCHANGE>:")
                x = torch.tensor(context, dtype=torch.long)[None, ...].to(trainer.device)
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]

                completion = train_dataset.decode_sequence(y.tolist())
                print(f"{completion = }")
            # save the latest model
            print("saving model")
            ckpt_path = os.path.join("models", "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

    return batch_end_callback


def main():
    model_config = GPT.get_default_config()
    trainer_config = Trainer.get_default_config()

    manglish = UnicodeData("data", model_config.block_size)

    model_config.model_type = 'gpt2-xl'
    model_config.vocab_size = manglish.get_vocab_size()
    model_config.block_size = 1024  # OpenAI's model block_size (i.e. input context length)
    model = GPT(model_config)

    trainer_config.batch_size = 1
    trainer_config.max_iters = 100_000
    trainer_config.num_workers = 6

    trainer = Trainer(trainer_config, model, manglish)
    trainer.set_callback('on_batch_end', get_batch_end_callback(model, manglish))

    trainer.run()


if __name__ == "__main__":
    main()

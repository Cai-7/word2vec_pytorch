from input_data import InputData
import numpy
from model import SkipGramModel, CBOWModel
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys


class Word2Vec:
    def __init__(self,
                 input_file_name,
                 output_file_name,
                 emb_dimension=300,
                 batch_size=50,
                 window_size=5,
                 iteration=1,
                 max_steps=None,
                 initial_lr=0.025,
                 min_count=5,
                 model_type='cbow',
                 checkpoint_dir='checkpoints',
                 checkpoint_interval=None,
                 subsample_t=1e-5,
                 resume_checkpoint=None):
        """Initilize class parameters.

        Args:
            input_file_name: Name of a text data from file. Each line is a sentence splited with space.
            output_file_name: Name of the final embedding file.
            emb_dimention: Embedding dimention, typically from 50 to 500.
            batch_size: The count of word pairs for one forward.
            window_size: Max skip length between words.
            iteration: Control the multiple training iterations.
            initial_lr: Initial learning rate.
            min_count: The minimal word frequency, words with lower frequency will be filtered.

        Returns:
            None.
        """
        self.data = InputData(input_file_name, min_count)
        self.data.subsample_t = subsample_t
        self.data.init_discard_table()
        self.output_file_name = output_file_name
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.window_size = window_size
        self.iteration = iteration
        self.max_steps = max_steps
        self.initial_lr = initial_lr
        self.min_count = min_count
        self.model_type = model_type.lower()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.subsample_t = subsample_t
        self.resume_checkpoint = resume_checkpoint
        self.start_step = 0
        if self.model_type == 'cbow':
            self.skip_gram_model = CBOWModel(self.emb_size, self.emb_dimension)
        else:
            self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.skip_gram_model.to(self.device)
        self.optimizer = optim.SGD(
            self.skip_gram_model.parameters(), lr=self.initial_lr)
        if self.resume_checkpoint:
            self.load_checkpoint(self.resume_checkpoint)

    def load_checkpoint(self, checkpoint_path):
        resolved_path = checkpoint_path
        if not os.path.isabs(resolved_path):
            candidate_path = os.path.join(self.checkpoint_dir, resolved_path)
            if os.path.exists(candidate_path):
                resolved_path = candidate_path

        if not os.path.exists(resolved_path):
            raise FileNotFoundError(
                'Checkpoint not found: %s (cwd=%s, checkpoint_dir=%s)' %
                (checkpoint_path, os.getcwd(), self.checkpoint_dir)
            )

        checkpoint = torch.load(resolved_path, map_location=self.device)
        self.skip_gram_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)

        self.start_step = int(checkpoint.get('step', 0))
        print('Loaded checkpoint: %s (step=%d)' % (resolved_path, self.start_step))

    def _get_checkpoint_interval(self, total_batches):
        if self.checkpoint_interval is not None:
            return self.checkpoint_interval
        if total_batches <= 1000:
            return max(1, total_batches)
        if total_batches <= 10000:
            return max(100, total_batches // 10)
        if total_batches <= 100000:
            return max(1000, total_batches // 20)
        if total_batches <= 1000000:
            return max(5000, total_batches // 50)
        return max(10000, total_batches // 100)

    def save_checkpoint(self, step, loss):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint_%08d.pt' % step)
        checkpoint = {
            'model_state_dict': self.skip_gram_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': step,
            'loss': loss,
            'emb_dimension': self.emb_dimension,
            'batch_size': self.batch_size,
            'window_size': self.window_size,
            'iteration': self.iteration,
            'initial_lr': self.initial_lr,
            'min_count': self.min_count,
            'model_type': self.model_type,
            'subsample_t': self.subsample_t,
            'input_file_name': self.data.input_file_name,
            'output_file_name': self.output_file_name,
        }
        torch.save(checkpoint, checkpoint_path)
        print('Checkpoint saved: %s' % checkpoint_path)

    def train(self):
        """Multiple training.

        Returns:
            None.
        """
        if self.model_type == 'cbow':
            pair_count = self.data.evaluate_cbow_count()
        else:
            pair_count = self.data.evaluate_pair_count(self.window_size)
        batch_count = int(self.iteration * pair_count / self.batch_size)
        if self.max_steps is not None and self.max_steps > 0:
            batch_count = min(batch_count, self.max_steps)
        if self.start_step >= batch_count:
            print('Resume step (%d) is already >= target steps (%d), no further training needed.' %
                  (self.start_step, batch_count))
            self.skip_gram_model.save_embedding(
                self.data.id2word, self.output_file_name, self.use_cuda)
            return
        checkpoint_interval = self._get_checkpoint_interval(batch_count)
        process_bar = tqdm(range(self.start_step, batch_count), dynamic_ncols=True)
        # self.skip_gram_model.save_embedding(
        #     self.data.id2word, 'begin_embedding.txt', self.use_cuda)
        for i in process_bar:
            if self.model_type == 'cbow':
                pos_u, pos_v = self.data.get_cbow_batch(self.batch_size,
                                                        self.window_size)
                neg_v = self.data.get_neg_v_neg_sampling(pos_v, 5)
                pos_u = torch.LongTensor(pos_u).to(self.device)
                pos_v = torch.LongTensor(pos_v).to(self.device)
            else:
                pos_pairs = self.data.get_batch_pairs(self.batch_size,
                                                      self.window_size)
                neg_v = self.data.get_neg_v_neg_sampling(pos_pairs, 5)
                pos_u = [pair[0] for pair in pos_pairs]
                pos_v = [pair[1] for pair in pos_pairs]

                pos_u = torch.LongTensor(pos_u).to(self.device)
                pos_v = torch.LongTensor(pos_v).to(self.device)
            neg_v = torch.LongTensor(neg_v).to(self.device)

            self.optimizer.zero_grad()
            loss = self.skip_gram_model(pos_u, pos_v, neg_v)
            loss.backward()
            self.optimizer.step()

            process_bar.set_description("Loss: %0.8f, lr: %0.6f" %
                                        (loss.item(),
                                         self.optimizer.param_groups[0]['lr']))
            if (i + 1) % checkpoint_interval == 0 or i + 1 == batch_count:
                self.save_checkpoint(i + 1, loss.item())
            if i * self.batch_size % 100000 == 0:
                lr = self.initial_lr * (1.0 - 1.0 * i / batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
        self.skip_gram_model.save_embedding(
            self.data.id2word, self.output_file_name, self.use_cuda)


if __name__ == '__main__':
    model_type = sys.argv[3] if len(sys.argv) > 3 else 'cbow'
    checkpoint_dir = sys.argv[4] if len(sys.argv) > 4 else 'checkpoints'
    max_steps = int(sys.argv[5]) if len(sys.argv) > 5 else None
    min_count = int(sys.argv[6]) if len(sys.argv) > 6 else 5
    subsample_t = float(sys.argv[7]) if len(sys.argv) > 7 else 1e-5
    resume_checkpoint = sys.argv[8] if len(sys.argv) > 8 and sys.argv[8] and sys.argv[8] != "none" else None
    emb_dimension = int(sys.argv[9]) if len(sys.argv) > 9 else 300
    iteration = int(sys.argv[10]) if len(sys.argv) > 10 else 1
    w2v = Word2Vec(input_file_name=sys.argv[1], output_file_name=sys.argv[2], model_type=model_type, checkpoint_dir=checkpoint_dir, max_steps=max_steps, min_count=min_count, subsample_t=subsample_t, resume_checkpoint=resume_checkpoint, emb_dimension=emb_dimension, iteration=iteration)
    w2v.train()

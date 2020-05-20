import torch
from transformers import AdamW, get_linear_schedule_with_warmup

class Trainer(object):

    def __init__(self,
                 option,
                 model=None,
                 train_dataloader=None,
                 dev_dataloader=None,
                 evaluator=None):


        self.option = option
        self.model = model
        self.train_dataloader =train_dataloader
        self.dev_dataloader = dev_dataloader
        self.evaluate = evaluator
        self.set_up_optimizer()

    def set_up_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        no_train = []
        if self.option['freeze'] == 'embedding':
            # freeze emb layer
            no_train.append('embedding')
        elif self.option['freeze'] != '0':
            layer = int(self.option['freeze'])
            no_train.append('embedding')
            for i in range(layer):
                no_train.append('layer.%d' % i)

        param_optimizer = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_train)]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters,
                               lr=self.option['learning_rate'],
                               correct_bias=False)


        num_train_examples = self.train_dataloader.size

        num_train_optimization_steps = int(num_train_examples / self.option['train_batchsize'] / self.option['gradient_accumulation_steps']) * self.option['max_epoch']

        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=int(self.option['warmup_proportion'] * num_train_optimization_steps),
                                                         num_training_steps=num_train_optimization_steps)


    def train(self):
        device = self.model.get_device()
        best_res = 0

        for epoch in range(self.option['max_epoch']):
            train_loss = 0
            self.model.train()
            for step, batch in enumerate(self.train_dataloader.dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, token_type_ids, attention_mask, valid_ids, pos_ids, ner_ids, deprel_ids, eval_idx, label_ids, label_mask, sen_id = batch #### ADD SRL ID #####
                loss, _ = self.model(input_ids,
                                        token_type_ids=token_type_ids,
                                        attention_mask=attention_mask,
                                        valid_ids=valid_ids,
                                        pos_ids=pos_ids,
                                        ner_ids=ner_ids,
                                        deprel_ids=deprel_ids,
                                        labels=label_ids,
                                        label_mask=label_mask)#### ADD SRL ID #####

                loss = loss / self.option['gradient_accumulation_steps']
                loss.backward()
                if (step + 1) % self.option['gradient_accumulation_steps'] == 0:

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                print('Epoch: %d, step: %d, training loss: %.5f' % (epoch, step, loss.item()))
                train_loss += loss.item()

            avg_train_loss = train_loss / len(self.train_dataloader)
            print("Epoch: %d, average training loss: %.5f" % (epoch, avg_train_loss))
            res, avg_dev_loss = self.evaluate(self.model, self.dev_dataloader)
            print("Epoch: %d, RES: %.5f, average dev loss: %.5f" % (epoch, res, avg_dev_loss))

            # Save best model
            if res > best_res:
                best_res = res
                # save best chpt
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                output_model_file = self.option['model_name']
                torch.save(model_to_save.state_dict(), output_model_file)

        self.model.load_state_dict(torch.load(self.option['model_name'], map_location=device))
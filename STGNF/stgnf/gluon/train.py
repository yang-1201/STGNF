from pts import Trainer
from pts.model import get_module_forward_input_names
from pts.dataset.loader import TransformedIterableDataset
from typing import List, Optional, Union
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from gluonts.core.component import validated
import time
import torch
from .utils import EarlyStopping
class STGNFTrainer(Trainer):
    def __init__(
            self,
            epochs: int = 100,
            train_batch_size: int = 32,
            val_batch_size:int=32,
            num_batches_per_epoch: int = 50,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-6,
            maximum_learning_rate: float = 1e-2,
            early_stop_patience: int=100,
            checkpoint_dir:str=None,
            clip_gradient: Optional[float] = None,
            device: Optional[Union[torch.device, str]] = None,
            **kwargs,
    ):
        super().__init__(epochs = epochs,
            batch_size = train_batch_size,
            num_batches_per_epoch = num_batches_per_epoch,
            learning_rate= learning_rate,
            weight_decay = weight_decay,
            maximum_learning_rate= maximum_learning_rate,
            clip_gradient = clip_gradient,
            device = device,
            **kwargs)
        print("TactisTrain")
        self.train_batch_size=train_batch_size
        self.val_batch_size=val_batch_size
        self.es = EarlyStopping(patience=early_stop_patience,
                           mode='min',  save_path=checkpoint_dir )
        self.checkpoint_dir=checkpoint_dir
        print("num_batches_per_epoch:", num_batches_per_epoch)
    def __call__(
        self,
        net: nn.Module,
        train_iter: DataLoader,
        validation_iter: Optional[DataLoader] = None,
    ) -> None:
        self.es.set_model(net)

        #print(validation_iter)
        optimizer = Adam(
            net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
            #amsgrad=True,
        )

        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=self.maximum_learning_rate,
            steps_per_epoch=self.num_batches_per_epoch,
            epochs=self.epochs,
        )
        for epoch_no in range(self.epochs):
            # mark epoch start time
            tic = time.time()
            cumm_epoch_loss = 0.0
            total = self.num_batches_per_epoch - 1

            # training loop
            with tqdm(train_iter, total=total,disable=True) as it:
            #with tqdm(train_iter, total=total) as it:
                print("******train epoch:{}/{}".format((epoch_no + 1) ,self.epochs),end='')
                for batch_no, data_entry in enumerate(it, start=1):
                    #print(data_entry)
                    optimizer.zero_grad()
                    inputs = [v.to(self.device) for v in data_entry.values()]
                    #{past_target_norm:[b,t1,n],future_target_norm:[b,t2,n]}
                    output = net(*inputs)

                    if isinstance(output, (list, tuple)):
                        loss = output[0]
                    else:
                        loss = output

                    cumm_epoch_loss += loss.item()
                    now_loss=loss.item()
                    avg_epoch_loss = cumm_epoch_loss / batch_no
                    it.set_postfix(
                        {
                            "epoch": f"{epoch_no + 1}/{self.epochs}",
                            "avg_loss": avg_epoch_loss,
                            "loss":now_loss,
                        },
                        refresh=False,
                    )

                    # for name, param in net.named_parameters():
                    #     print('name:{} param grad:{} param requires_grad:{}'.format(name, param.grad,
                    #                                                                 param.requires_grad))
                    # torch.autograd.set_detect_anomaly(True)
                    #
                    # with torch.autograd.detect_anomaly():
                    loss.backward(retain_graph=True)

                    #net.parameters()
                    # # 访问每个参数的梯度
                    # for param in net.parameters():
                    #    print(param.grad)
                    #
                    # # 更新模型参数
                    # net.trainer.step(1)
                    # # 访问每个参数的梯度
                    # print(net.weight.grad)
                    # print(net.bias.grad)

                    #loss.backward()
                    # if self.clip_gradient is not None:
                    #     nn.utils.clip_grad_norm_(net.parameters(), self.clip_gradient)
                    #torch.nn.utils.clip_grad_value_(net.parameters(), -0.1, 10)
                    # for param in net.parameters():
                    #     param.data.clamp_(min=-0.0000000001, max=5)

                    optimizer.step()
                    lr_scheduler.step()

                    if self.num_batches_per_epoch == batch_no:
                        break
                it.close()
                print(", avg_loss: %.2f, "% avg_epoch_loss+"loss: %.2f"%now_loss)

            # validation loop
            if validation_iter is not None:
               # print(validation_iter)
                print("val epoch: {}/{}".format(epoch_no + 1,self.epochs), end='')
                cumm_epoch_loss_val = 0.0
                #early_stop=False
                #with tqdm(validation_iter, total=total, colour="green") as it:
                with tqdm(validation_iter, total=total, colour="green",disable=True) as it:

                    for batch_no, data_entry in enumerate(it, start=1):
                        inputs = [v.to(self.device) for v in data_entry.values()]
                        with torch.no_grad():
                            output = net(*inputs)
                        if isinstance(output, (list, tuple)):
                            loss = output[0]
                        else:
                            loss = output

                        cumm_epoch_loss_val += loss.item()
                        now_loss=loss.item()
                        avg_epoch_loss_val = cumm_epoch_loss_val / batch_no
                        #print(avg_epoch_loss_val)
                        it.set_postfix(
                            {
                                "epoch": f"{epoch_no + 1}/{self.epochs}",
                                #"avg_loss": avg_epoch_loss,
                                "avg_val_loss": avg_epoch_loss_val,
                                "loss":now_loss,
                            },
                            refresh=False,
                        )

                        if self.num_batches_per_epoch == batch_no:
                            break
                        # if self.es.step(now_loss, epoch_no + 1):
                        #     early_stop=True
                        #     print('early stopped! With val loss:', now_loss)
                        #     print('best_epoch:{}'.format(self.es.get_best_epoch()))
                        #     break
                it.close()

                print(", avg_val_loss: %.2f"%avg_epoch_loss_val+", loss: %.2f"%now_loss)

                if self.es.step(avg_epoch_loss_val, epoch_no + 1):
                    print('early stopped! With val loss:', avg_epoch_loss_val)
                    print('best_epoch:{}'.format(self.es.get_best_epoch()))
                    break

                # if early_stop:
                #     break
            # mark epoch end time and log time cost of current epoch
            toc = time.time()
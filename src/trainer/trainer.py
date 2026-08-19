import torch
from torch.utils.data import DataLoader

from src.utils.meter import AverageMeter



class Trainer:
    """
    Training manager

    Dataset
        |
        DataLoader
        |
        Model
        |
        Loss
        |
        Optimizer

    """


    def __init__(
        self,
        model,
        dataset,
        loss_fn,
        optimizer,
        device="cpu",
        batch_size=1,
        num_workers=0
    ):


        self.device=device


        self.model=model.to(
            device
        )


        self.loss_fn=loss_fn


        self.optimizer=optimizer



        self.loader=DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )



        self.meter=AverageMeter()



    def train_epoch(
        self
    ):


        self.model.train()


        self.meter.reset()


        for batch in self.loader:


            feature=batch["feature"].to(
                self.device
            )


            state=batch["state"].to(
                self.device
            )


            ##################################
            # forward
            ##################################

            output=self.model(
                feature
            )


            ##################################
            # prediction extraction
            ##################################

            if isinstance(output,dict):

                prediction=output["state"]

            else:

                prediction=output



            ##################################
            # loss
            ##################################

            loss=self.loss_fn(
                prediction,
                state
            )



            ##################################
            # backward
            ##################################

            self.optimizer.zero_grad()


            loss.backward()


            self.optimizer.step()



            self.meter.update(
                loss.item()
            )



        return self.meter.avg



    def fit(
        self,
        epochs
    ):


        history=[]


        for epoch in range(
            epochs
        ):


            loss=self.train_epoch()


            history.append(
                loss
            )


            print(
                f"Epoch [{epoch+1}/{epochs}] Loss:{loss:.6f}"
            )


        return history

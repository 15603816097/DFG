import os
import time
import torch

import numpy as np

from torch.utils.data import DataLoader, random_split

from src.utils.meter import AverageMeter



class Trainer:


    def __init__(
        self,
        model,
        dataset,
        loss_fn,
        optimizer,
        device="cpu",
        batch_size=16,
        num_workers=0,
        save_dir="checkpoints"
    ):


        self.device=device


        self.model=model.to(
            device
        )


        self.loss_fn=loss_fn


        self.optimizer=optimizer


        self.save_dir=save_dir


        os.makedirs(
            save_dir,
            exist_ok=True
        )



        ####################################
        # dataset split
        ####################################


        total=len(dataset)


        train_len=int(
            total*0.7
        )


        val_len=int(
            total*0.15
        )


        test_len=(
            total
            -
            train_len
            -
            val_len
        )


        self.train_set, self.val_set, self.test_set = random_split(

            dataset,

            [
                train_len,
                val_len,
                test_len
            ],

            generator=torch.Generator().manual_seed(42)

        )



        self.train_loader=DataLoader(

            self.train_set,

            batch_size=batch_size,

            shuffle=True,

            num_workers=num_workers

        )


        self.val_loader=DataLoader(

            self.val_set,

            batch_size=batch_size,

            shuffle=False,

            num_workers=num_workers

        )


        self.test_loader=DataLoader(

            self.test_set,

            batch_size=batch_size,

            shuffle=False,

            num_workers=num_workers

        )



        print(
            "Train:",
            len(self.train_set),
            "Val:",
            len(self.val_set),
            "Test:",
            len(self.test_set)
        )



    ########################################
    # train
    ########################################


    def train_epoch(self):


        self.model.train()


        meter=AverageMeter()



        for batch in self.train_loader:


            feature=batch["feature"].to(
                self.device
            )


            state=batch["state"].to(
                self.device
            )


            output=self.model(
                feature
            )


            prediction=output["state"]



            loss=self.loss_fn(

                prediction,

                state

            )



            self.optimizer.zero_grad()


            loss.backward()


            self.optimizer.step()



            meter.update(
                loss.item()
            )



        return meter.avg



    ########################################
    # validation
    ########################################


    @torch.no_grad()

    def evaluate(
        self,
        loader
    ):


        self.model.eval()



        preds=[]

        targets=[]



        for batch in loader:


            feature=batch["feature"].to(
                self.device
            )


            state=batch["state"].to(
                self.device
            )



            output=self.model(
                feature
            )


            prediction=output["state"]



            preds.append(
                prediction.cpu()
            )


            targets.append(
                state.cpu()
            )



        preds=torch.cat(
            preds,
            dim=0
        )


        targets=torch.cat(
            targets,
            dim=0
        )



        error=preds-targets



        rmse=torch.sqrt(

            torch.mean(
                error**2
            )

        ).item()



        mae=torch.mean(

            torch.abs(
                error
            )

        ).item()



        return rmse,mae



    ########################################
    # test
    ########################################


    def test(self):


        rmse,mae=self.evaluate(

            self.test_loader

        )


        print(
            "\nTest Result:"
        )


        print(
            f"RMSE:{rmse:.6f}"
        )


        print(
            f"MAE:{mae:.6f}"
        )



        return rmse,mae




    ########################################
    # full training
    ########################################


    def fit(
        self,
        epochs
    ):


        history=[]


        best_rmse=float(
            "inf"
        )



        log_file=os.path.join(

            self.save_dir,

            "train_log.txt"

        )


        with open(
            log_file,
            "w"
        ) as f:



            for epoch in range(
                epochs
            ):



                start=time.time()



                train_loss=self.train_epoch()



                val_rmse,val_mae=self.evaluate(

                    self.val_loader

                )



                history.append(

                    train_loss

                )



                msg=(

                    f"Epoch {epoch+1}/{epochs} "

                    f"Loss={train_loss:.6f} "

                    f"Val_RMSE={val_rmse:.6f} "

                    f"Val_MAE={val_mae:.6f} "

                    f"Time={time.time()-start:.2f}s"

                )



                print(
                    msg
                )


                f.write(
                    msg+"\n"
                )



                ################################
                # best model
                ################################


                if val_rmse < best_rmse:


                    best_rmse=val_rmse


                    torch.save(

                        self.model.state_dict(),

                        os.path.join(

                            self.save_dir,

                            "best_model.pth"

                        )

                    )


                    print(
                        "Best model saved"
                    )



            torch.save(

                self.model.state_dict(),

                os.path.join(

                    self.save_dir,

                    "last_model.pth"

                )

            )



        return history

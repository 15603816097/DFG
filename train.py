import os

import torch
import yaml

from torch.optim import Adam


from src.dataset.kitti_real_dataset import KITTIRawDataset

from src.model.model_pipeline import ModelPipeline

from src.loss.state_loss import StateLoss

from src.trainer import Trainer



def load_config(path):

    with open(
        path,
        "r"
    ) as f:

        return yaml.safe_load(f)




def plot_loss(history):

    import matplotlib.pyplot as plt


    os.makedirs(
        "results",
        exist_ok=True
    )


    plt.figure()


    plt.plot(
        history
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Loss"
    )


    plt.title(
        "Training Loss"
    )


    plt.savefig(

        "results/loss_curve.png"

    )


    plt.close()




def main():


    config=load_config(

        "configs/train.yaml"

    )



    device=torch.device(

        "cuda"

        if torch.cuda.is_available()

        else

        "cpu"

    )



    print(
        "Device:",
        device
    )



    dataset=KITTIRawDataset(

        config["dataset"]["root"]

    )



    print(

        "Dataset length:",

        len(dataset)

    )




    model=ModelPipeline(

        input_dim=51,

        state_dim=6

    )




    criterion=StateLoss()




    optimizer=Adam(

        model.parameters(),

        lr=config["train"]["lr"]

    )





    trainer=Trainer(

        model=model,

        dataset=dataset,

        loss_fn=criterion,

        optimizer=optimizer,

        device=device,

        batch_size=config["train"]["batch_size"],

        save_dir=config["checkpoint"]["save_dir"]

    )





    history=trainer.fit(

        config["train"]["epochs"]

    )



    plot_loss(

        history

    )



    trainer.test()



    print(

        "Training finished"

    )




if __name__=="__main__":

    main()

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

        config=yaml.safe_load(f)

    return config




def main():


    ####################################
    # config
    ####################################

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



    ####################################
    # Dataset
    ####################################


    dataset=KITTIRawDataset(
        config["dataset"]["root"]
    )


    print(
        "Dataset length:",
        len(dataset)
    )



    ####################################
    # Model
    ####################################


    model=ModelPipeline(
        input_dim=51,
        state_dim=6
    )



    ####################################
    # Loss
    ####################################


    criterion=StateLoss()



    ####################################
    # Optimizer
    ####################################


    optimizer=Adam(
        model.parameters(),
        lr=config["train"]["lr"]
    )



    ####################################
    # Trainer
    ####################################


    trainer=Trainer(

        model=model,

        dataset=dataset,

        loss_fn=criterion,

        optimizer=optimizer,

        device=device,

        batch_size=config["train"]["batch_size"]

    )



    ####################################
    # Train
    ####################################


    epochs=config["train"]["epochs"]


    for epoch in range(
        epochs
    ):


        loss=trainer.train_epoch()


        print(
            f"Epoch {epoch+1}/{epochs}, Loss={loss:.6f}"
        )



    ####################################
    # Save
    ####################################


    torch.save(

        model.state_dict(),

        "checkpoints/model.pth"

    )


    print(
        "Training finished"
    )




if __name__=="__main__":


    main()

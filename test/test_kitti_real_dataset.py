import os


from src.dataset.kitti_real_dataset import (
    KITTIRawDataset,
)



DATASET_ROOT=(

    "dataset/kitti/"
    "2011_10_03/"
    "2011_10_03_drive_0027_sync"

)





def test_dataset_length():


    dataset=KITTIRawDataset(

        DATASET_ROOT

    )


    assert len(dataset)>0





def test_dataset_output():


    dataset=KITTIRawDataset(

        DATASET_ROOT

    )


    sample=dataset[0]



    assert sample["feature"].shape[0]==20


    assert sample["feature"].shape[1]==51



    assert sample["state"].shape==(6,)





def test_dataset_no_nan():


    dataset=KITTIRawDataset(

        DATASET_ROOT

    )


    sample=dataset[0]



    assert not sample["feature"].isnan().any()



    assert not sample["state"].isnan().any()

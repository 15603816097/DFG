from src.loader.gps_loader import GPSLoader



DATASET_ROOT = (
    "dataset/kitti/"
    "2011_10_03/"
    "2011_10_03_drive_0027_sync"
)



def test_gps():

    loader = GPSLoader(
        DATASET_ROOT
    )


    gps = loader[0]


    print(gps)


    assert "position" in gps
    assert "velocity" in gps

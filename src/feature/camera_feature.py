import cv2
import numpy as np



class CameraFeatureExtractor:
    """
    Camera image quality feature extractor.


    Output:

        8 dimensional feature

    """



    def extract(
        self,
        image,
    ):


        image=np.asarray(
            image
        )


        if image is None:

            raise ValueError(
                "Image is None"
            )


        gray=cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )


        brightness=np.mean(
            gray
        )


        contrast=np.std(
            gray
        )


        sharpness=cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()


        edges=cv2.Canny(
            gray,
            100,
            200
        )


        edge_ratio=np.mean(
            edges>0
        )


        feature=np.array(
            [
                brightness,

                contrast,

                sharpness,

                edge_ratio,

                np.min(gray),

                np.max(gray),

                np.mean(
                    gray<30
                ),

                np.mean(
                    gray>220
                ),
            ],
            dtype=np.float64
        )


        return feature

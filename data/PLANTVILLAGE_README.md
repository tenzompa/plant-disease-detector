# PlantVillage download instructions

The CV block uses the [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset).

1. Install the Kaggle CLI:  `pip install kaggle`
2. Place your `~/.kaggle/kaggle.json` API token.
3. Run:

    ```bash
    mkdir -p data/plantvillage
    cd data/plantvillage
    kaggle datasets download -d abdallahalidev/plantvillage-dataset
    unzip plantvillage-dataset.zip
    ```

4. The training script (`train_cv_model.py`) expects the directory
   layout `data/plantvillage/color/<class_name>/*.jpg`.
5. We use the 15 classes listed in `data/dataset_metadata.json`.

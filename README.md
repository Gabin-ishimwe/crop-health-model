# crop-health-model

This file goes over a few useful things to know about how to configure the entire training process including the data and the model, as well as how to run the training and evaluation processes. Finally, instructions are given on how to generate a model archive (MAR) file that can be provided to TorchServe.

## Main model configurations

To train a model, start by configuring the data, model and training through the `config_<type>.yaml` files inside the `configs` folder. Currently, three main model configurations are defined. 

### Binary classifier

The first model is a binary classifier, predicting whether a crop is healthy or sick. In this configuration, the following fields are defined in the following way:
````
fit.model.model.init_args.num_classes: 2
fit.data.task: binary
````
Additionally, all loggers in `fit.trainer.logger` have `name` set to `binary`.

### Multiclass classifier with a single healthy class

The second model is a multiclass classifier with a single healthy class, predicting whether a crop is healthy or afflicted with one of several diseases. In this configuration, the following fields are defined in the following way:
````
fit.model.model.init_args.num_classes: 9
fit.data.task: single-HLT
````
Additionally, all loggers in `fit.trainer.logger` have `name` set to `single_HLT`.

### Multiclass classifier with multiple healthy classes

The third model is a multiclass classifier with several healthy classes (one for each crop type), predicting whether a crop of a specific type is healthy or afflicted with one of several diseases. In this configuration, the following fields are defined in the following way:
````
fit.model.model.init_args.num_classes: 11
fit.data.task: multi-HLT
````
Additionally, all loggers in `fit.trainer.logger` have `name` set to `multi_HLT`.

### One model per crop (Team 8)

We train a separate classifier for each crop in our scope, using `config_cassava.yaml` (HLT, CMD, CBSD), `config_maize.yaml` (HLT, FAW, MSV, MLB, MLN) and `config_beans.yaml` (HLT, ALS, BR). They use `fit.data.crops` to keep only that crop's images, stronger augmentation (random crops, rotation, colour changes), and a single device (`accelerator: auto` picks the Mac GPU or a Colab GPU).

## Prepare the data

One command gets the data into `.data`, ready for training. Run it from the root of the project, pointing it to the shared zip in the team's Google Drive folder (on Colab, Drive is mounted at `/content/drive/MyDrive`; on a Mac with Google Drive for desktop, at `~/Library/CloudStorage/GoogleDrive-<email>/My Drive`):
```
python -m crop_health_model.data.make_dataset --drive-zip "<drive folder>/crop-health/crop-data.zip"
```
- If the zip exists, it is copied to local disk and extracted. This is the normal case and takes a few minutes. Running it again does nothing if the same zip is already extracted.
- If the zip does not exist yet, the datasets are downloaded from Harvard Dataverse one at a time, the images are kept with their shorter side at 512 pixels, each archive is deleted after extraction, `annotations.csv` is generated, and everything is packed into the zip for the rest of the team. Each finished dataset is also saved next to the zip (in `crop-data-parts/`), so if the run is interrupted, running the same command again restores those and only downloads the rest. The parts are deleted once the full zip is written.

The first full download is easiest on Colab, which downloads much faster than a home connection: open [notebooks/prepare_data_colab.ipynb](https://colab.research.google.com/github/Gabin-ishimwe/crop-health-model/blob/main/notebooks/prepare_data_colab.ipynb) and run all cells.

Without `--drive-zip`, the data is only downloaded and annotated locally. To try the pipeline quickly, download only the smallest dataset (1.9 GB): `python -m crop_health_model.data.make_dataset --datasets maize-dataset-namibia`. The data is shared as one zip because reading thousands of small images directly from Google Drive is very slow.

## Additional configuration

A lot of additional elements within the data, the model and training process can be configured.

### Data configuration

Several aspects of the data can be configured. For example:
- The train, validation and test proportions are set through `fit.data.data_split`, for example `[0.7, 0.15, 0.15]`. Each class is split separately with the same proportions, and `fit.data.split_seed` keeps the split identical between training and testing.
- `fit.data.test_sources` can be set to a list of dataset folders (for example `["maize-dataset-namibia"]`). All images from those datasets then form the test set and are never seen in training, which shows how the model does on photos from a new source.
- `fit.data.crops` keeps only the images of the listed crop types.
- The number of workers to use in the `DataLoader` through `fit.data.num_workers`.
- The transforms to apply to the training and test/validation data through `fit.data.train_transforms` and `fit.data.test_transforms`, respectively.
- The normalization to apply to apply to the data (which will be the same for both the training and test/validation data) through `fit.data.normalization`.

During a development stage, the `fit.data.limit` can be set to a positive integer, such as `1000`, to use a random sample of 1000 images.

### Model configuration

The PyTorch model itself can be configured by specifying the class path and any potential initialization arguments. For example, to use the custom `ResNet` model, we set `fit.model.mode.class_path` to `crop_health_model.models.model.ResNet`. For this model, we can also specify the use of pre-trained weights by setting `weights` to `DEFAULT`, the number of layers by setting `num_layers` to (for example) `18`, and the number of output features by setting `num_classes` to (for example) `2`.

Currently, only a ResNet model has been defined for use in this project, but other models can be easily added. Note that if more models are added, they should be split into their respective files, such as `resnet.py`, `densenet.py`, etc.. inside the `models` folder. This is because torch-model-archiver needs to receive a script containing a single `nn.Module` corresponding to the appropriate model defintion.

### Training configuration

The training itself is also highly configurable. In our case, we set `fit.trainer.deterministic` to `true` in order to have reproducible experiments. We also set the maximum number of epochs to train for through `fit.train.max_epochs`. Which device(s) to use can also be configured through `fit.trainer.devices`. Its default value is `auto`, but it can for example be set to `[0,1]` if the training should be done on two GPUs.

Existing PyTorch Lightning callbacks as well as custom callbacks can be added through `fit.training.callbacks`. Loggers can also be added through `fit.train.logger`.

The optimizer and learning rate scheduler can be set through the `fit.optimizer` and `fit.lr_scheduler` fields.

## Launch training and evaluation

To run a training process, run the following command from the root of this project:
```
python3 crop_health_model/scripts/train.py --config crop_health_model/configs/config_multi_HLT.yaml fit
```

For a quick check that everything runs, add `--data.limit 1000 --trainer.max_epochs 1`.

To evaluate a model on the validation set, identify the model version and the filename of its checkpoint, then run the following from the root of this project:
```
python3 crop_health_model/scripts/train.py validate --config tb_logs/multi_HLT/version_0/config.yaml --ckpt_path tb_logs/multi_HLT/version_0/checkpoints/crop_health_model-epoch=5-step=7260-val_loss=0.181.ckpt --trainer.devices=1
```
Here we set the number of devices to be 1 to avoid getting a warning about using distributed dataloaders in a validation/test phase. Such a value could also be set directly in the config file provided through the `--config` argument.

Similarly, to evaluate the above model on the test set, run the following. Besides accuracy and F1, the test run prints the recall and precision of each class and saves `test_per_class.csv`, `test_confusion_matrix.csv` and `test_confusion_matrix.png` in the new log folder it creates:
```
python3 crop_health_model/scripts/train.py test --config tb_logs/multi_HLT/version_0/config.yaml --ckpt_path tb_logs/multi_HLT/version_0/checkpoints/crop_health_model-epoch=5-step=7260-val_loss=0.181.ckpt --trainer.devices=1
```

## Create a model archive (MAR) file

To create a model archive file to be used by TorchServe, simply navigate to the folder of the specific model and version (in this case single-HLT version 2 -> `tb_logs/single_HLT/version_2`) to archive and run:
```
torch-model-archiver --model-name single-HLT --version 2.0 --model-file model_script.py --serialized-file checkpoints/best_model.pt --handler model_handler.py --extra-files index_to_name.json
```
For this command, only the `--model-name` and `--version` need to be adjsuted for each case.

Torch Model Archiver will create a `single-HLT.mar` file inside `tb_logs/single_HLT/version_2`. To deploy this model, first add it to the `model_store` folder inside `torch_serve` in the [crop-health-api](https://github.com/openearthplatforminitiative/crop-health-api) repository.
Then add a mapping from name to filename in the `torch_serve/docker/config.properties` file, under the `load_models` field.

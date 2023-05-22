# TODO:
'''
- CAPIRE COSA FARE E COME!!

'''



print("this is train.py")

############################  IMPORT ############################
import argparse
from pyannote.database.protocol.protocol import Preprocessor
from pyannote.database import ProtocolFile, FileFinder, registry
from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio.tasks import VoiceActivityDetection
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from types import MethodType
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning.strategies.ddp import DDPStrategy
import time
from datetime import datetime

############################  IMPORT ############################


############################  ARGS ############################
# Create an argument parser
parser = argparse.ArgumentParser(description='Parsing the args of the python script...')

# Add the command-line arguments
parser.add_argument('--encoder', type=str, help='Encoder parameter')
parser.add_argument('--decoder', type=str, help='Decoder parameter')

# Parse the arguments
args = parser.parse_args()

# Access the encoder and decoder values
encoder = args.encoder
decoder = args.decoder

# Print the encoder and decoder values
print(f'Encoder: {encoder}')
print(f'Decoder: {decoder}')
############################  ARGS ############################

############################  INITIALIZATIONS ############################
path_to_database_file = '/om2/user/fabiocat/csad/data/pyannote/mini_new/database.yml'
my_protocol_name = "MyChildrenDataset.SpeakerDiarization.real"

# GENERAL
optimizer_name = "Adam"
max_epochs = 2  # >>>> 10 # TODO: TO CHANGE!!
window_duration = 5.0  # from hpt
lr = 0.0006  # 0.00044610980058861524

# ECAPA
# none

# HuBERT
# none

# Wav2Vec2.0
# none

# Wav2vec2.0 child
# none

#SINCNET
my_stride = 1
my_stride = 10 ** my_stride

#LSTM
lstm_num_layers = 2

#TRANSFORMER
#...

#GPT2
#...

############################  INITIALIZATIONS ############################

############################  BODY ############################
start_time = time.time()
print("start_time")
print(start_time)

# pyannote protocol
registry.load_database(path_to_database_file)


class MyLabelMapper(Preprocessor):
    def __call__(self, current_file: ProtocolFile):
        labels = current_file["annotation"].labels()
        mapping = {lbl: lbl for lbl in labels}
        annotations = current_file["annotation"].rename_labels(mapping=mapping)
        annotations = annotations.subset(set(['child']))
        return annotations


dataset = registry.get_protocol(my_protocol_name,
                                {"audio": FileFinder(),
                                 "annotation": MyLabelMapper()})


def configure_optimizers(self):
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    return {"optimizer": optimizer,
            "lr_scheduler": ExponentialLR(optimizer, 0.9)}


vad = VoiceActivityDetection(dataset, num_workers=64, duration=window_duration, batch_size=128, pin_memory=True)

model = PyanNet(task=vad, sincnet={"stride": my_stride},
                lstm={"hidden_size": 128, "num_layers": lstm_num_layers, "bidirectional": True, "monolithic": True,
                      "dropout": 0.5}, linear={"hidden_size": 128, "num_layers": 2})

'''
for name, param in model.named_parameters():
    if name.startswith('ss_model'):
        param.requires_grad = False

for name, param in model.named_parameters():
    if param.requires_grad:
        # print(name)
        pass
'''

model.configure_optimizers = MethodType(configure_optimizers, model)

value_to_monitor_2, min_or_max_2 = vad.val_monitor
#value_to_monitor_2 = "VoiceActivityDetection-MyChildrenDatasetSpeakerDiarizationreal-TrainLoss"  # work-around
#min_or_max_2 = "min"  # work-around

value_to_monitor = value_to_monitor_2

now = datetime.now()
model_checkpoint = ModelCheckpoint(
    monitor=value_to_monitor_2,
    mode=min_or_max_2,
    save_top_k=2,
    every_n_epochs=1,
    save_last=True,
    dirpath="./torch/checkpoints/" + now.strftime("%d_%m_%Y__%H_%M_%S") + "/",
    filename=f"{{epoch}}-{{{value_to_monitor}:.6f}}",
    verbose=True)

early_stopping = EarlyStopping(
    monitor=value_to_monitor_2,
    mode=min_or_max_2,
    min_delta=0.01,
    patience=10.,
    strict=True,
    verbose=True)

logger = TensorBoardLogger("./torch/logger/" + now.strftime("%d_%m_%Y__%H_%M_%S") + "/", name="", version="",
                           log_graph=False)

trainer = pl.Trainer(accelerator='gpu',
                     max_epochs=max_epochs,
                     strategy=DDPStrategy(find_unused_parameters=True),
                     callbacks=[model_checkpoint, early_stopping],
                     logger=logger,
                     )

trainer.fit(model=model)
print("--- %s seconds ---" % (time.time() - start_time))
############################  BODY ############################

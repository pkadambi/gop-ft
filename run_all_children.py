import glob
import argparse
from src.DataprepStages import PrepareFeaturesAndModelsStage, AlignCrossValStage, AlignHeldoutStage, \
    CreateLabelsCrossValStage, CreateLabelsHeldoutStage
from src.GopStages import GopHeldoutStage, EvaluateGopStage, GopStage
from src.utils.run_utils import CreateExperimentDirectoryStage
from src.Config import DataprepConfig, GopConfig
from src.Stages import ComplexStage

def run_on_speaker(speakerid):
    config_dict = DataprepConfig('./configs/dataprep_child.yaml', speakerid).config_dict

    prep_stage = PrepareFeaturesAndModelsStage(config_dict)
    align_stage = ComplexStage([AlignCrossValStage(config_dict),
                                AlignHeldoutStage(config_dict)], 'align')
    data_stages = [prep_stage, align_stage]
    data_pipeline = ComplexStage(data_stages, 'dataprep')
    data_pipeline.run()

    config_dict = GopConfig('./configs/gop_kaldi_labels.yaml', use_heldout=True, speakerid=speakerid).config_dict
    prepdir_stage = CreateExperimentDirectoryStage(config_dict)
    gop_stage = GopHeldoutStage(config_dict)
    gop_stages = [prepdir_stage, gop_stage]
    gop_pipeline = ComplexStage(gop_stages, 'data_gop_pipeline')
    gop_pipeline.run()

def run_all_children(speakerids):
    for speakerid in speakerids:
        run_on_speaker(speakerid)

if __name__=='__main__':
    config_file = './configs/dataprep.yaml'
    speakerids = [folder.split('/')[-1] for folder in glob.glob('./child_speech_16_khz_test/*') if '.txt' not in folder]

    run_all_children(speakerids)

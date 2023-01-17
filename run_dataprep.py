import argparse
from src.DataprepStages import PrepareFeaturesAndModelsStage, AlignCrossValStage, AlignHeldoutStage, \
    CreateLabelsCrossValStage, CreateLabelsHeldoutStage
from src.Config import DataprepConfig
from src.Stages import ComplexStage


def run_all(config_yaml, speakerid):
    config_dict = DataprepConfig(config_yaml, speakerid).config_dict

    prep_stage = PrepareFeaturesAndModelsStage(config_dict)
    align_stage = ComplexStage([AlignCrossValStage(config_dict), AlignHeldoutStage(config_dict)], "align")
    labels_stage = ComplexStage([CreateLabelsCrossValStage(config_dict),
                                 CreateLabelsHeldoutStage(config_dict)],
                                "labels")

    dataprep_stages = [prep_stage, align_stage, labels_stage]
    # dataprep_stages = [prep_stage, align_stage]

    dataprep = ComplexStage(dataprep_stages, "dataprep")

    dataprep.run()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', dest='config_yaml',  help='Path .yaml config file for experiment', default=None)
    #
    # args = parser.parse_args()
    config_file = './configs/dataprep_child.yaml'
    speakerid = '0611_F_AP'
    # config_file = './configs/speakers/6011_F_AP'
    run_all(config_file, speakerid)
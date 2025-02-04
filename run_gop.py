import argparse
from src.utils.run_utils import CreateExperimentDirectoryStage
from src.GopStages import GopHeldoutStage, EvaluateGopStage, GopStage
from src.Config import GopConfig
from src.Stages import ComplexStage
from IPython import embed

def run_all(config_yaml, from_stage, to_stage, use_heldout, holdout_speaker):

    config_dict = GopConfig(config_yaml, use_heldout, holdout_speaker).config_dict

    prepdir_stage = CreateExperimentDirectoryStage(config_dict)
    gop_stage     = GopHeldoutStage(config_dict) if use_heldout else GopStage(config_dict)
    eval_stage    = EvaluateGopStage(config_dict)

    gop_pipeline  = ComplexStage([prepdir_stage, gop_stage, eval_stage], "gop-pipeline")

    gop_pipeline.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_yaml',  help='Path .yaml config file for experiment', default=None)
    parser.add_argument('--from', dest='from_stage',  help='First stage to run (prepdir, gop, evaluate)', default=None)
    parser.add_argument('--to', dest='to_stage',  help='Last stage to run (prepdir, gop, evaluate)', default=None)
    parser.add_argument('--heldout', action='store_true', help='Use this option to test on heldout set', default=False)

    args = parser.parse_args()
    use_heldout = args.heldout
    use_heldout = True

    args.from_stage = 'gop'
    args.to_stage = 'evaluate'

    args.config_yaml='./configs/gop_kaldi_labels.yaml'
    holdout_speaker = '0611_F_AP'
    run_all(args.config_yaml, args.from_stage, args.to_stage, use_heldout, holdout_speaker)
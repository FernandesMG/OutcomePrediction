import subprocess
import sys
import shlex


def run(cmd):
    print(f">>> {cmd}", flush=True)
    # shell=False is safer; use a list for args
    proc = subprocess.run(shlex.split(cmd), check=True)
    return proc.returncode


steps = [
    r'python /home/ann/Code/OutcomePrediction/Classification.py --set RUN.cv_fold=0 RUN.max_epochs=1000 RUN.train_fraction=1.0 MODEL.learning_rate=5e-3 MODEL.dropout_prob=0.5 DATA.model_name="SGDMCoxWCensoringFullCroppedRes" MODEL.lr_warmup_epochs=4',
    r'python /home/ann/Code/OutcomePrediction/Classification.py --set RUN.cv_fold=1 RUN.max_epochs=1000 RUN.train_fraction=1.0 MODEL.learning_rate=5e-3 MODEL.dropout_prob=0.5 DATA.model_name="SGDMCoxWCensoringFullCroppedRes" MODEL.lr_warmup_epochs=4',
    r'python /home/ann/Code/OutcomePrediction/Classification.py --set RUN.cv_fold=0 RUN.max_epochs=1000 RUN.train_fraction=1.0 MODEL.learning_rate=1e-3 MODEL.dropout_prob=0.5 DATA.model_name="SGDMCoxWCensoringFullCroppedRes" MODEL.lr_warmup_epochs=4',
    r'python /home/ann/Code/OutcomePrediction/Classification.py --set RUN.cv_fold=1 RUN.max_epochs=1000 RUN.train_fraction=1.0 MODEL.learning_rate=1e-3 MODEL.dropout_prob=0.5 DATA.model_name="SGDMCoxWCensoringFullCroppedRes" MODEL.lr_warmup_epochs=4',
    r'python /home/ann/Code/OutcomePrediction/Classification.py --set RUN.cv_fold=0 RUN.max_epochs=1000 RUN.train_fraction=1.0 MODEL.learning_rate=1e-2 MODEL.dropout_prob=0.5 DATA.model_name="SGDMCoxWCensoringFullCroppedRes" MODEL.lr_warmup_epochs=4',
    r'python /home/ann/Code/OutcomePrediction/Classification.py --set RUN.cv_fold=1 RUN.max_epochs=1000 RUN.train_fraction=1.0 MODEL.learning_rate=1e-2 MODEL.dropout_prob=0.5 DATA.model_name="SGDMCoxWCensoringFullCroppedRes" MODEL.lr_warmup_epochs=4',
]

# steps = [
#     r'python /home/ann/Code/OutcomePrediction/Classification.py --set RUN.cv_fold=0 RUN.max_epochs=600 RUN.train_fraction=1.0 MODEL.learning_rate=1e-4 MODEL.dropout_prob=0.5 DATA.model_name="MuonCoxWCensoringFullCroppedRes" MODEL.optimizer="Muon" MODEL.opt_params="dict()"',
#     r'python /home/ann/Code/OutcomePrediction/Classification.py --set RUN.cv_fold=1 RUN.max_epochs=600 RUN.train_fraction=1.0 MODEL.learning_rate=1e-4 MODEL.dropout_prob=0.5 DATA.model_name="MuonCoxWCensoringFullCroppedRes" MODEL.optimizer="Muon" MODEL.opt_params="dict()"',
#     r'python /home/ann/Code/OutcomePrediction/Classification.py --set RUN.cv_fold=2 RUN.max_epochs=600 RUN.train_fraction=1.0 MODEL.learning_rate=1e-4 MODEL.dropout_prob=0.5 DATA.model_name="MuonCoxWCensoringFullCroppedRes" MODEL.optimizer="Muon" MODEL.opt_params="dict()"',
#     r'python /home/ann/Code/OutcomePrediction/Classification.py --set RUN.cv_fold=3 RUN.max_epochs=600 RUN.train_fraction=1.0 MODEL.learning_rate=1e-4 MODEL.dropout_prob=0.5 DATA.model_name="MuonCoxWCensoringFullCroppedRes" MODEL.optimizer="Muon" MODEL.opt_params="dict()"',
#     r'python /home/ann/Code/OutcomePrediction/Classification.py --set RUN.cv_fold=4 RUN.max_epochs=600 RUN.train_fraction=1.0 MODEL.learning_rate=1e-4 MODEL.dropout_prob=0.5 DATA.model_name="MuonCoxWCensoringFullCroppedRes" MODEL.optimizer="Muon" MODEL.opt_params="dict()"',
# ]

try:
    for s in steps:
        run(s)
    print("All steps completed.")
except subprocess.CalledProcessError as e:
    print(f"Step failed with exit code {e.returncode}: {e.cmd}", file=sys.stderr)
    sys.exit(e.returncode)

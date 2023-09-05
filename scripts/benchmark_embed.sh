#!/bin/sh

usage="Script to qsub jobs to create VAE or GroupEnc embedding(s) of a dataset saved as an .npy file

$(basename "$0") [-h] [-d dataset] [-n dataset_name] [-l target_dim] [-k group_size] [-r fpath_res] [-g fpath_groupenc] [-o fpath_logs] [-c fname_script] [-v fname_venv] [-m memory] [-e epochs] [-s runs]

where:
    -h --help           show this help message and exit
    -d --dataset        path to .npy file with row-wise coordinates of input data
    -n --dataset_name   short name of the input dataset
    -l --target_dim     target dimensionality of the latent space, and the resulting embedding (default: 2)
    -k --group_size     group size (gamma) hyperparameter (0 ~ VAE; 3+ ~ GroupEnc) (default: 4)
    -r --fpath_res      path to where result will be saved (default: './res')
    -g --fpath_groupenc path to the GroupEnc package (default: './GroupEnc')
    -o --fpath_logs     path to where error and output logs will be written (default: './logs')
    -c --fname_script   path to embed.py script (default: './embed.py')
    -v --fname_venv     path to an existing venv (virtual environment) with GroupEnc dependencies (default: './venv_embed')
    -m --memory         RAM required for each job (single model training) in GB (default: 16)
    -e --epochs         number of training epochs (default: 500)
    -s --runs           number of repeated runs with different random seeds (default: 5)
"

LATENT_DIM=2
GROUP_SIZE=4
FPATH_RES="./res"
FPATH_GROUPENC="./GroupEnc"
FPATH_LOGS="./logs"
FNAME_SCRIPT="./embed.py"
FPATH_VENV="./venv_embed"
MEMORY=16
EPOCHS=500
RUNS=5

optspec=":h:d:n:k:l:r:g:o:c:v:m:e:s:-:"
while getopts "$optspec" optchar;
do
    case "${optchar}" in
	-)
		case "${OPTARG}" in
			help) echo "$usage"
			      exit
			      ;;
			dataset) DATASET=${OPTARG};;
			dataset_name) DATASET_NAME=${OPTARG};;
			group_size) GROUP_SIZE=${OPTARG};;
			target_dim) LATENT_DIM=${OPTARG};;
			fpath_res) FPATH_RES=${OPTARG};;
			fpath_groupenc) FPATH_GROUPENC=${OPTARG};;
			fpath_logs) FPATH_LOGS=${OPTARG};;
			fname_script) FNAME_SCRIPT=${OPTARG};;
			fname_venv) FNAME_VENV=${OPTARG};;
			memory) MEMORY=${OPTARG};;
			epochs) EPOCHS=${OPTARG};;
			runs) RUNS=${OPTARG};;
		esac;;
	h) echo "$usage"
       	   exit
           ;;
       	d) DATASET=${OPTARG};;
	n) DATASET_NAME=${OPTARG};;
	k) GROUP_SIZE=${OPTARG};;
	l) LATENT_DIM=${OPTARG};;
	r) FPATH_RES=${OPTARG};;
	g) FPATH_GROUPENC=${OPTARG};;
	o) FPATH_LOGS=${OPTARG};;
	c) FNAME_SCRIPT=${OPTARG};;
	v) FNAME_VENV=${OPTARG};;
	m) MEMORY=${OPTARG};;
	e) EPOCHS=${OPTARG};;
	s) RUNS=${OPTARG};;
    esac
done

for i in $(seq $RUNS)
do
	n="${DATASET_NAME}k${GROUP_SIZE}l${LATENT_DIM}i${i}"
	er="${FPATH_LOGS}/${n}Embed_ERROR.txt"
	ou="${FPATH_LOGS}/${n}Embed_OUTPUT.txt"
	echo "qsubbing ${n}"
	cat <<EOS | qsub -l nodes=1:ppn=quarter:gpus=1,mem=${MEMORY}gb,walltime=0:30:00 -e $er -o $ou -N $n
source ${FPATH_VENV}/bin/activate
mkdir -p $FPATH_RES
python $FNAME_SCRIPT -f $DATASET -d $DATASET_NAME -r $FPATH_RES -k $GROUP_SIZE -l $LATENT_DIM -e $EPOCHS -i $i -s $FPATH_GROUPENC
EOS
done


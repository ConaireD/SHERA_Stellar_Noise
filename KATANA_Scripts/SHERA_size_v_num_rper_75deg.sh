#PBS -o ./output_reports
#
#PBS -l select=1:ncpus=8:mem=2gb
#PBS -l walltime=4:00:00
#PBS -J 0-625

module load python/3.10.8

source ~/.venvs/venv-tutorial-1/bin/activate

cd /srv/scratch/z5459921/SHERA
python3 -u SHERA_size_v_num_rper_75deg.py ./Label_Run_files/${PBS_ARRAY_INDEX}.npy


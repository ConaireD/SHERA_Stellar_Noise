#PBS -o ./output_reports
#
#PBS -l select=1:ncpus=8:mem=4gb
#PBS -l walltime=11:59:00
#PBS -J 0-8000

module load python/3.10.8

source ~/.venvs/venv-tutorial-1/bin/activate

cd /srv/scratch/z5459921/SHERA
python3 -u SHERA_size_v_num_v_obs_phi_robust_new.py ${PBS_ARRAY_INDEX}
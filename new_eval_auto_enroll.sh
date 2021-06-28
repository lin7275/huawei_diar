# you need to change the paths in libsph_cohort.tsv
python3   diar_with_id_densenet.py -m densenet.tar  -t trials.tsv -c libsph_cohort.tsv
python3   diar_with_id_new_resnet.py -m resnet.tar -t trials.tsv -c libsph_cohort.tsv
python3 fusion.py
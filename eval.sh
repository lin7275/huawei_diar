# eval diarization on replay_music set
python3 -W ignore  diar.py --model_file model_final_319 --corpus_dir diar_music
# eval diarization with speaker id
python3 -W ignore  diar_with_id.py -m model_final_319 -t manual_enroll_list_for_disk_diar_fixed.tsv
# eval diarization with auto enrollment
python3 -W ignore diar_with_id.py -m model_final_319 -t final_auto_enroll_libsph.tsv
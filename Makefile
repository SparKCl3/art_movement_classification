#################### PACKAGE ACTIONS ###################
#reinstall_package:
	#@pip uninstall -y taxifare || :
	#@pip install -e .

#################### PREPROCESS ###################
run_preprocess:
	python -c "from art_movement_classification.art_classification.art_modelling.main import preprocessing; preprocessing('train'); preprocessing('test'); preprocessing('valid')"

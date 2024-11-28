###################### .env #####################
PYTHON := python3
SRC_DIR := src
ENV_FILE := .env

############## Default variables ################
export BATCH_SIZE ?= 32
export NUM_CLASSES ?= 26
export EPOCHS ?= 10
export PATIENCE ?= 3
export LEARNING_RATE ?= 0.001
export LOCAL_REGISTRY_PATH ?= models

#################### Targets ###################
.PHONY: all setup train evaluate save load clean help

#################### All ###################
all: train evaluate

#################### Setup ###################
setup:
	@echo "Setting up environment..."
	@pip install -r requirements.txt
	@if [ -f $(ENV_FILE) ]; then export $(shell sed 's/#.*//g' $(ENV_FILE) | xargs); fi
	@echo "✅ Environment setup complete."

#################### Train ###################
train:
	@echo "Training the model..."
	$(PYTHON) $(SRC_DIR)/main.py --train
	@echo "✅ Training complete."

#################### Evaluate ###################
evaluate:
	@echo "Evaluating the model..."
	$(PYTHON) $(SRC_DIR)/main.py --evaluate
	@echo "✅ Evaluation complete."

#################### Save ###################
save:
	@echo "Saving the model..."
	$(PYTHON) $(SRC_DIR)/main.py --save
	@echo "✅ Model saved."

#################### Load ###################
load:
	@echo "Loading the model..."
	$(PYTHON) $(SRC_DIR)/main.py --load
	@echo "✅ Model loaded."

######## Remove generated files ############
clean:
	@echo "Cleaning up..."
	@rm -rf $(LOCAL_REGISTRY_PATH)/* models/cp.ckpt
	@echo "✅ Cleanup complete."

################## Commands ###############
help:
	@echo "Available targets:"
	@echo "  setup      - Install dependencies and prepare environment."
	@echo "  train      - Train the model."
	@echo "  evaluate   - Evaluate the model."
	@echo "  save       - Save the trained model."
	@echo "  load       - Load the most recent saved model."
	@echo "  clean      - Remove generated files (models, checkpoints)."
	@echo "  help       - Display this help message."

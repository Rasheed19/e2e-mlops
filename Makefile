.PHONY: setup
setup: install create-env

.PHONY: install
install:
	@echo "Installing dependencies..."
	pip install --upgrade pip &&\
	pip install -r requirements.txt

.PHONY: create-env
create-env:
	@echo "Creating .env file..."
	@read -p "Enter ROLE: " ROLE; \
	read -p "Enter S3_BUCKET_NAME: " S3_BUCKET_NAME; \
	echo "ROLE=$$ROLE" > .env; \
	echo "S3_BUCKET_NAME=$$S3_BUCKET_NAME" >> .env
	@echo ".env file created successfully."

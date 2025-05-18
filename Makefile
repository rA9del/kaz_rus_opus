IMAGE_NAME=adel
DATA_PATH=/hdd/adel

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run --gpus all -it --rm \
        -v $(PWD):/home \
        -v $(DATA_PATH):/dataset \
        $(IMAGE_NAME) /bin/bash

clean:
	docker rmi $(IMAGE_NAME)

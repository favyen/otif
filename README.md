OTIF
----

This is the implementation of the OTIF fast object tracking approach proposed in "OTIF: Efficient Tracker Pre-processing over Large Video Datasets" (SIGMOD 2022).

See our website at https://favyen.com/otif/ for more details.


Dataset
-------

First, download and extract the dataset:

	wget https://favyen.com/files/otif-dataset.zip
	unzip otif-dataset.zip

The dataset consists of these components:

- raw_video/: original video files from which the dataset is made.
- yolov3/: object detector configuration files, weights, and training data.
- dataset/: the actual dataset.


Installation
------------

Clone this repository:

	git clone https://github.com/favyen/otif

Also clone and build darknet-alexey (YOLOv3 object detector):

	cd /path/to/otif-dataset/
	git clone https://github.com/AlexeyAB/darknet darknet-alexey/
	cd darknet-alexey/
	git checkout ecad770071eb776208a46977347e6d2410d4f50e
	make

Setup conda environment:

	conda create -n otif python=3.6
	conda activate otif
	pip install scikit-image 'tensorflow<2.0'

Install Go, ffmpeg:

	sudo apt install golang ffmpeg


Run Experiments
---------------

The provided dataset includes the trained segmentation proxy and tracker models, so we can run parameter tuning directly. In a later section we will see how to train these models.

Run parameter tuning for each dataset, using both our method and the baselines:

	cd otif/pipeline2
	export PYTHONPATH=../python
	go run 30_speed_accuracy.go /path/to/otif-dataset/ amsterdam ours-simple yolov3-640x352 416_256_-1 '{"Thresholds":[0,0,1,1,1,1,1]}' 10 amsterdam-valid-ours-simple.txt
	go run 30_speed_accuracy.go /path/to/otif-dataset/ amsterdam chameleon-good yolov3-640x352 416_256_-1 '{"Thresholds":[0,0,1,1,1,1,1]}' 10 amsterdam-valid-chameleon.txt
	go run 30_speed_accuracy.go /path/to/otif-dataset/ amsterdam blazeit yolov3-640x352 416_256_-1 '{"Thresholds":[0,0,1,1,1,1,1]}' 10 amsterdam-valid-blazeit.txt
	go run 30_speed_accuracy.go /path/to/otif-dataset/ amsterdam miris yolov3-640x352 416_256_-1 '{"Thresholds":[0,0,1,1,1,1,1]}' 10 amsterdam-valid-miris.txt
	go run 30_speed_accuracy.go /path/to/otif-dataset/ jackson ours-simple yolov3-736x416 416_256_-1 '{"Thresholds":[0,0,1,1,1,1,1]}' 10 jackson-valid-ours-simple.txt
	go run 30_speed_accuracy.go /path/to/otif-dataset/ jackson chameleon-good yolov3-736x416 416_256_-1 '{"Thresholds":[0,0,1,1,1,1,1]}' 10 jackson-valid-chameleon.txt
	go run 30_speed_accuracy.go /path/to/otif-dataset/ jackson blazeit yolov3-736x416 416_256_-1 '{"Thresholds":[0,0,1,1,1,1,1]}' 10 jackson-valid-blazeit.txt
	go run 30_speed_accuracy.go /path/to/otif-dataset/ jackson miris yolov3-736x416 416_256_-1 '{"Thresholds":[0,0,1,1,1,1,1]}' 10 jackson-valid-miris.txt
	go run 30_speed_accuracy.go /path/to/otif-dataset/ shibuya ours-simple yolov3-960x544 416_256_-1 '{"Thresholds":[0,1,1,1,1,1,1]}' 10 shibuya-valid-ours-simple.txt
	go run 30_speed_accuracy.go /path/to/otif-dataset/ shibuya chameleon-good yolov3-960x544 416_256_-1 '{"Thresholds":[0,1,1,1,1,1,1]}' 10 shibuya-valid-chameleon.txt
	go run 30_speed_accuracy.go /path/to/otif-dataset/ shibuya blazeit yolov3-960x544 416_256_-1 '{"Thresholds":[0,1,1,1,1,1,1]}' 10 shibuya-valid-blazeit.txt
	go run 30_speed_accuracy.go /path/to/otif-dataset/ shibuya miris yolov3-960x544 416_256_-1 '{"Thresholds":[0,1,1,1,1,1,1]}' 10 shibuya-valid-miris.txt
	go run 30_speed_accuracy.go /path/to/otif-dataset/ warsaw ours-simple yolov3-960x544 416_256_-1 '{"Thresholds":[0,1,1,1,1,1,1]}' 10 warsaw-valid-ours-simple.txt
	go run 30_speed_accuracy.go /path/to/otif-dataset/ warsaw chameleon-good yolov3-960x544 416_256_-1 '{"Thresholds":[0,1,1,1,1,1,1]}' 10 warsaw-valid-chameleon.txt
	go run 30_speed_accuracy.go /path/to/otif-dataset/ warsaw blazeit yolov3-960x544 416_256_-1 '{"Thresholds":[0,1,1,1,1,1,1]}' 10 warsaw-valid-blazeit.txt
	go run 30_speed_accuracy.go /path/to/otif-dataset/ warsaw miris yolov3-960x544 416_256_-1 '{"Thresholds":[0,1,1,1,1,1,1]}' 10 warsaw-valid-miris.txt
	go run 30_speed_accuracy.go /path/to/otif-dataset/ uav ours-simple yolov3-960x512 416_256_-1 '{"Thresholds":[0,1,1,1,1,1,1]}' 10 uav-valid-ours-simple.txt
	go run 30_speed_accuracy.go /path/to/otif-dataset/ uav chameleon-good yolov3-960x512 416_256_-1 '{"Thresholds":[0,1,1,1,1,1,1]}' 10 uav-valid-chameleon.txt
	go run 30_speed_accuracy.go /path/to/otif-dataset/ uav blazeit yolov3-960x512 416_256_-1 '{"Thresholds":[0,1,1,1,1,1,1]}' 10 uav-valid-blazeit.txt
	go run 30_speed_accuracy.go /path/to/otif-dataset/ uav miris yolov3-960x512 416_256_-1 '{"Thresholds":[0,1,1,1,1,1,1]}' 10 uav-valid-miris.txt
	go run 30_speed_accuracy.go /path/to/otif-dataset/ caldot1 ours-simple yolov3-704x480 416_256_-1 '{"Thresholds":[0,1,1,1,1,1,1]}' 10 caldot1-valid-ours-simple.txt
	go run 30_speed_accuracy.go /path/to/otif-dataset/ caldot1 chameleon-good yolov3-704x480 416_256_-1 '{"Thresholds":[0,1,1,1,1,1,1]}' 10 caldot1-valid-chameleon.txt
	go run 30_speed_accuracy.go /path/to/otif-dataset/ caldot1 blazeit yolov3-704x480 416_256_-1 '{"Thresholds":[0,1,1,1,1,1,1]}' 10 caldot1-valid-blazeit.txt
	go run 30_speed_accuracy.go /path/to/otif-dataset/ caldot1 miris yolov3-704x480 416_256_-1 '{"Thresholds":[0,1,1,1,1,1,1]}' 10 caldot1-valid-miris.txt
	go run 30_speed_accuracy.go /path/to/otif-dataset/ caldot2 ours-simple yolov3-704x480 416_256_-1 '{"Thresholds":[0,1,1,1,1,1,1]}' 10 caldot2-valid-ours-simple.txt
	go run 30_speed_accuracy.go /path/to/otif-dataset/ caldot2 chameleon-good yolov3-704x480 416_256_-1 '{"Thresholds":[0,1,1,1,1,1,1]}' 10 caldot2-valid-chameleon.txt
	go run 30_speed_accuracy.go /path/to/otif-dataset/ caldot2 blazeit yolov3-704x480 416_256_-1 '{"Thresholds":[0,1,1,1,1,1,1]}' 10 caldot2-valid-blazeit.txt
	go run 30_speed_accuracy.go /path/to/otif-dataset/ caldot2 miris yolov3-704x480 416_256_-1 '{"Thresholds":[0,1,1,1,1,1,1]}' 10 caldot2-valid-miris.txt

Then apply the best parameters at different accuracy and speed levels on the test set:

	go run 40_evaltest.go /path/to/otif-dataset/ amsterdam ours-simple amsterdam-valid-ours-simple.txt amsterdam-test-ours-simple.txt
	go run 40_evaltest.go /path/to/otif-dataset/ amsterdam chameleon-good amsterdam-valid-chameleon.txt amsterdam-test-chameleon.txt
	go run 40_evaltest.go /path/to/otif-dataset/ amsterdam blazeit amsterdam-valid-blazeit.txt amsterdam-test-blazeit.txt
	go run 40_evaltest.go /path/to/otif-dataset/ amsterdam miris amsterdam-valid-miris.txt amsterdam-test-miris.txt
	go run 40_evaltest.go /path/to/otif-dataset/ jackson ours-simple jackson-valid-ours-simple.txt jackson-test-ours-simple.txt
	go run 40_evaltest.go /path/to/otif-dataset/ jackson chameleon-good jackson-valid-chameleon.txt jackson-test-chameleon.txt
	go run 40_evaltest.go /path/to/otif-dataset/ jackson blazeit jackson-valid-blazeit.txt jackson-test-blazeit.txt
	go run 40_evaltest.go /path/to/otif-dataset/ jackson miris jackson-valid-miris.txt jackson-test-miris.txt
	go run 40_evaltest.go /path/to/otif-dataset/ shibuya ours-simple shibuya-valid-ours-simple.txt shibuya-test-ours-simple.txt
	go run 40_evaltest.go /path/to/otif-dataset/ shibuya chameleon-good shibuya-valid-chameleon.txt shibuya-test-chameleon.txt
	go run 40_evaltest.go /path/to/otif-dataset/ shibuya blazeit shibuya-valid-blazeit.txt shibuya-test-blazeit.txt
	go run 40_evaltest.go /path/to/otif-dataset/ shibuya miris shibuya-valid-miris.txt shibuya-test-miris.txt
	go run 40_evaltest.go /path/to/otif-dataset/ warsaw ours-simple warsaw-valid-ours-simple.txt warsaw-test-ours-simple.txt
	go run 40_evaltest.go /path/to/otif-dataset/ warsaw chameleon-good warsaw-valid-chameleon.txt warsaw-test-chameleon.txt
	go run 40_evaltest.go /path/to/otif-dataset/ warsaw blazeit warsaw-valid-blazeit.txt warsaw-test-blazeit.txt
	go run 40_evaltest.go /path/to/otif-dataset/ warsaw miris warsaw-valid-miris.txt warsaw-test-miris.txt
	go run 40_evaltest.go /path/to/otif-dataset/ uav ours-simple uav-valid-ours-simple.txt uav-test-ours-simple.txt
	go run 40_evaltest.go /path/to/otif-dataset/ uav chameleon-good uav-valid-chameleon.txt uav-test-chameleon.txt
	go run 40_evaltest.go /path/to/otif-dataset/ uav blazeit uav-valid-blazeit.txt uav-test-blazeit.txt
	go run 40_evaltest.go /path/to/otif-dataset/ uav miris uav-valid-miris.txt uav-test-miris.txt
	go run 40_evaltest.go /path/to/otif-dataset/ caldot1 ours-simple caldot1-valid-ours-simple.txt caldot1-test-ours-simple.txt
	go run 40_evaltest.go /path/to/otif-dataset/ caldot1 chameleon-good caldot1-valid-chameleon.txt caldot1-test-chameleon.txt
	go run 40_evaltest.go /path/to/otif-dataset/ caldot1 blazeit caldot1-valid-blazeit.txt caldot1-test-blazeit.txt
	go run 40_evaltest.go /path/to/otif-dataset/ caldot1 miris caldot1-valid-miris.txt caldot1-test-miris.txt
	go run 40_evaltest.go /path/to/otif-dataset/ caldot2 ours-simple caldot2-valid-ours-simple.txt caldot2-test-ours-simple.txt
	go run 40_evaltest.go /path/to/otif-dataset/ caldot2 chameleon-good caldot2-valid-chameleon.txt caldot2-test-chameleon.txt
	go run 40_evaltest.go /path/to/otif-dataset/ caldot2 blazeit caldot2-valid-blazeit.txt caldot2-test-blazeit.txt
	go run 40_evaltest.go /path/to/otif-dataset/ caldot2 miris caldot2-valid-miris.txt caldot2-test-miris.txt

Each text file contains a speed-accuracy curve: the first columns show the parameter configuration used, and the last three columns show runtime, validation accuracy, and test accuracy.


Frame-Level Queries
-------------------

Above, we reproduced the track query experiments. We can also run the frame-level query experiments to compare against BlazeIt and TASTI.

	# OTIF: run with configuration yielding accuracy within 5% of the best-achieved accuracy
	cd otif/pipeline2
	mkdir /tmp/tracks/
	go run cmd_exec.go /path/to/otif-dataset/ shibuya ours-simple tracker /tmp/tracks/ '{"Name":"yolov3","Dims":[640,352],"Sizes":[[640,352],[448,128],[192,64]],"Threshold":0.25}' '416_256_0.04096' '{"NumFrames":0,"Thresholds":[0,0,0,0,1,1,1]}'
	cd otif/blazeit2/
	python ours_get_outputs.py /path/to/otif-dataset/ shibuya /tmp/tracks/ ./output/
	# BlazeIt
	cd otif/blazeit2/
	python prepare_blazeit2_train.py /path/to/otif-dataset/ shibuya yolov3-960x544
	python train.py /path/to/otif-dataset/dataset/shibuya/train/blazeit2-train/images/ 64 64 /path/to/otif-dataset/dataset/shibuya/blazeit2_model/64x64/
	python apply_on_video.py /path/to/otif-dataset/ shibuya 64 64 /path/to/otif-dataset/dataset/shibuya/blazeit2_model/64x64/ /path/to/otif-dataset/dataset/shibuya/blazeit2_model/scores.json
	python get_query_frames.py /path/to/otif-dataset/ shibuya yolov3-960x544 ./output/
	# TASTI
	cd otif/tasti/
	python tasti/examples/otif.py /path/to/otif-dataset/ shibuya ./outputs/shibuya/tasti_frames.json
	cd otif/blazeit2/
	python tasti_get_query_frames.py /path/to/otif-dataset/ shibuya yolov3-960x544 ./output/

Pre-processing
--------------

Pre-processing is only needed to setup the dataset from scratch from the raw video files.

First, extract several one-minute segments of video for the training, validation, and test sets:

	python data/raw_to_dataset/caldot_one_minute.py /path/to/otif-dataset/
	python data/raw_to_dataset/miris_one_minute.py /path/to/otif-dataset/
	python data/raw_to_dataset/one_minute_samples.py /path/to/otif-dataset/
	python data/raw_to_dataset/uav_one_minute.py /path/to/otif-dataset/

Second, train the YOLOv3 models. Due to how YOLOv3 configuration files are setup, you will need to change paths in train.txt, valid.txt, test.txt, and obj.data. Models should be trained at each of several resolutions, and for each dataset. Amsterdam and Jackson do not have training data, so we use the model pre-trained on COCO. For example:

	cd /path/to/otif-dataset/
	cd darknet-alexey
	./darknet detector train ../yolov3/shibuya/obj.data ../yolov3/shibuya/yolov3-640x352.cfg darknet53.conv.74

Third, identify the best object detector resolution. To do so, run the object detector at each available resolution, and compare the performance. For the Tokyo dataset:

	cd otif/pipeline2/
	ls /path/to/otif-dataset/yolov3/shibuya/ | grep best # see what detectors are available
	go run 10_detect.go /path/to/otif-dataset/ shibuya /path/to/otif-dataset/dataset/shibuya/valid/video/ /path/to/otif-dataset/dataset/shibuya/valid/yolov3-1280x704/ 1280x704
	go run 10_detect.go /path/to/otif-dataset/ shibuya /path/to/otif-dataset/dataset/shibuya/valid/video/ /path/to/otif-dataset/dataset/shibuya/valid/yolov3-960x544/ 960x544
	go run 10_detect.go /path/to/otif-dataset/ shibuya /path/to/otif-dataset/dataset/shibuya/valid/video/ /path/to/otif-dataset/dataset/shibuya/valid/yolov3-736x416/ 736x416
	go run 10_detect.go /path/to/otif-dataset/ shibuya /path/to/otif-dataset/dataset/shibuya/valid/video/ /path/to/otif-dataset/dataset/shibuya/valid/yolov3-640x352/ 640x352
	go run 10_detect.go /path/to/otif-dataset/ shibuya /path/to/otif-dataset/dataset/shibuya/valid/video/ /path/to/otif-dataset/dataset/shibuya/valid/yolov3-576x320/ 576x320
	go run iou-tracker2.go /path/to/otif-dataset/ shibuya /path/to/otif-dataset/dataset/shibuya/valid/video/ /path/to/otif-dataset/dataset/shibuya/valid/yolov3-1280x704/ ./1280x704_tracks/ iou '' '{"Thresholds":[0,1,1,1,1,1,1]}'
	go run iou-tracker2.go /path/to/otif-dataset/ shibuya /path/to/otif-dataset/dataset/shibuya/valid/video/ /path/to/otif-dataset/dataset/shibuya/valid/yolov3-960x544/ ./960x544_tracks/ iou '' '{"Thresholds":[0,1,1,1,1,1,1]}'
	go run iou-tracker2.go /path/to/otif-dataset/ shibuya /path/to/otif-dataset/dataset/shibuya/valid/video/ /path/to/otif-dataset/dataset/shibuya/valid/yolov3-736x416/ ./736x416_tracks/ iou '' '{"Thresholds":[0,1,1,1,1,1,1]}'
	go run iou-tracker2.go /path/to/otif-dataset/ shibuya /path/to/otif-dataset/dataset/shibuya/valid/video/ /path/to/otif-dataset/dataset/shibuya/valid/yolov3-640x352/ ./640x352_tracks/ iou '' '{"Thresholds":[0,1,1,1,1,1,1]}'
	go run iou-tracker2.go /path/to/otif-dataset/ shibuya /path/to/otif-dataset/dataset/shibuya/valid/video/ /path/to/otif-dataset/dataset/shibuya/valid/yolov3-576x320/ ./576x320_tracks/ iou '' '{"Thresholds":[0,1,1,1,1,1,1]}'
	go run eval.go /path/to/otif-dataset/ shibuya valid ./1280x704_tracks/
	go run eval.go /path/to/otif-dataset/ shibuya valid ./960x544_tracks/
	go run eval.go /path/to/otif-dataset/ shibuya valid ./736x416_tracks/
	go run eval.go /path/to/otif-dataset/ shibuya valid ./640x352_tracks/
	go run eval.go /path/to/otif-dataset/ shibuya valid ./576x320_tracks/

Fourth, apply the best detector on training and tracker data.

	go run 10_detect.go /path/to/otif-dataset/ shibuya /path/to/otif-dataset/dataset/shibuya/train/video/ /path/to/otif-dataset/dataset/shibuya/train/yolov3-960x544/ 960x544
	go run 10_detect.go /path/to/otif-dataset/ shibuya /path/to/otif-dataset/dataset/shibuya/tracker/video/ /path/to/otif-dataset/dataset/shibuya/tracker/yolov3-960x544/ 960x544
	go run iou-tracker2.go /path/to/otif-dataset/ shibuya /path/to/otif-dataset/dataset/shibuya/tracker/video/ /path/to/otif-dataset/dataset/shibuya/tracker/yolov3-960x544/ /path/to/otif-dataset/dataset/shibuya/tracker/tracks/ iou '' '{"Thresholds":[0,1,1,1,1,1,1]}'

Fifth, extract video frames for training the OTIF and baseline proxy models, and train the proxy and tracker models.

	cd otif/pipeline/
	python 16_extract_frames.py /path/to/otif-dataset/ shibuya yolov3-960x544
	python 16_blazeit_frames.py /path/to/otif-dataset/ shibuya yolov3-960x544
	python 20_train_segmentation.py /path/to/otif-dataset/ shibuya car
	python 20_train_blazeit.py /path/to/otif-dataset/ shibuya
	cd otif/rnn/
	python train.py /path/to/otif-dataset/ shibuya
	cd otif/tracker-miris/
	python train.py /path/to/otif-dataset/ shibuya yolov3-960x544

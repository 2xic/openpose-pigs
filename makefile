
clean:
	find . -name '*.pyc' -exec rm {} +
	find . -name '*.pyo' -exec rm {} +
	rm -rf ./build/
	rm -rf ./dist/
	rm -rf *.egg-info

install: clean
	pip install -r requirements.txt
	python3.8 setup.py bdist_wheel --universal
	python3.8 -m pip install ./dist/openpose-0.0.2-py2.py3-none-any.whl

visualize: install
	cd openpose/scripts && python3 visualize.py --image-name=N861D6_ch5_main_20200909140000_20200909143000.mp4_frame_15999.jpg
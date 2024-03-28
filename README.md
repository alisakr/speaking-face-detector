Ever wanted to know who speaker 10 is on a transcript? This is tool that allows you to find the face of each speaker in a transcript. In partnership with [Reduct](https://reduct.video), you can pass in any mp4 and get a photo of the face of every speaker. [Reduct](https://reduct.video) is a collaborative transcript-based video and audio platform for reviewing, searching, highlighting, and editing content of people talking, at scale.

For setup after downloading this repository run the following steps on command line assuming pip is installed. Use "deactivate" to exit the project specific python environment.
1. python3 -m venv env
2. source env/bin/activate
3. python -m pip install --upgrade pip
4. pip install -r requirements.txt

The above steps are working locally for package authors on python version 3.9.1. You can check last_successful_run_pip_list.txt to see what all package versions were on the last run of package scripts that we attempted.

For predicting the faces of every paragraph, you can use the script get_predictions.py, like the example below, in the debug_output_directory we print out a frame and view of speaker face for every frame where a speaking face is found. In the output_directory, we save the most likely face of each paragraph's speaker.

python scripts/get_predictions.py --api_key_yaml api_key.yaml --video_path "../../Downloads/input_video.mp4" --debug_output_directory debug_output --output_directory final_output
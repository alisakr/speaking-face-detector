Ever wanted to know who speaker 10 is on a transcript? This is tool that allows you to find the face of each speaker in a transcript. In partnership with [Reduct](https://reduct.video), you can pass in any mp4 and get a photo of the face of every speaker. [Reduct](https://reduct.video) is a collaborative transcript-based video and audio platform for reviewing, searching, highlighting, and editing content of people talking, at scale.

For setup after downloading this repository run the following steps on command line assuming pip is installed. Use "deactivate" to exit the project specific python environment.
1. `python3.9 -m venv env`
2. `source env/bin/activate`
3. `python -m pip install --upgrade pip`
4. `pip install -r requirements.txt`

The above steps are working locally for package authors on python version 3.9.1. You can check last_successful_run_pip_list.txt to see what all package versions were on the last run of package scripts that we attempted.

For predicting the faces of every words, you can use the script get_predictions_from_image_model_only.py,  In the output_directory, we save the faces found for every word and in the output json we save likely face of each paragraph's speaker.

`python scripts/get_predictions.py --api_key_yaml my_api_key.yaml --video_path "../../Downloads/input_video.mp4" --output_directory output_dir --output_json output.json`

the api_key yaml should include the following 2 lines

`reduct_api_key: myapikey`
`reduct_organization_id: organizationId_or_email`
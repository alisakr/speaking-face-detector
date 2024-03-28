from datetime import datetime
import json
import random
import string
import time
import yaml

from constants import (
    max_wait_for_transcript_seconds,
    reduct_api_key,
    reduct_organization_key,
    reduct_transcript_complete_status,
)
from gateways.reduct import (
    create_doc_from_file,
    create_project,
    get_transcript_status,
    get_transcript,
)


def get_or_create_transcript(args):
    # get or create transcript from command line args
    api_key = None
    reduct_organization_id = None
    if args.video_path == "":
        raise Exception("No video path provided")
    transcript_json = None
    if args.input_transcript != "":
        transcript_json = None
        with open(args.input_transcript, "r") as f:
            transcript_json = f.read()
            transcript_json = json.loads(transcript_json)
        return transcript_json
    if args.api_key_yaml:
        api_file = open(args.api_key_yaml, "r")
        api_config = yaml.safe_load(api_file)
        api_file.close()
        api_key = api_config.get(reduct_api_key, None)
        reduct_organization_id = api_config.get(reduct_organization_key, None)
    if api_key is None:
        raise Exception("No api key provided")
    if args.doc_id == "" and reduct_organization_id is None:
        raise Exception("No doc_id or reduct_organization_id provided")
    elif args.doc_id == "":
        # create project then create doc then upload video
        random_title_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        # use date as the prefix
        title_prefix = datetime.utcnow().strftime(format="%Y_%m_%d_%H_%M_%S_%f")[:-3]
        object_name = f"{title_prefix}_{random_title_suffix}"
        response = create_project(object_name, reduct_organization_id, api_key_override=api_key)
        project_id = response['id'] 
        response = create_doc_from_file(project_id, object_name, args.video_path, api_key_override=api_key)
        doc_id = response['doc_id']
    else:
        doc_id = args.doc_id
    start_time = datetime.utcnow()
    while True:
        transcript_status = get_transcript_status(doc_id, api_key_override=api_key)
        if len(transcript_status['media']) == 0:
            raise Exception("No media found in transcript status")
        # assume only one media item
        status = next(iter(transcript_status['media'].values()))
        if status == reduct_transcript_complete_status:
            break
        if (datetime.utcnow() - start_time).total_seconds() > max_wait_for_transcript_seconds:
            raise Exception("Transcript took too long to transcribe")
        #TODO: get the failed status of the transcript
        print(f"Transcript not ready yet, status={status}, waiting 5 seconds")
        # sleep for 5 seconds
        time.sleep(5)
    # get the transcript now that it is transcribed
    return get_transcript(doc_id, api_key_override=api_key)
    

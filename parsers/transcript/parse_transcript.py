import json

def get_times_and_speakers(json_filename, speakers_to_include=None, start_time_seconds=0, end_time_seconds=None, image_for_each_segment=True):
    '''
    Given a json filename, returns a list of tuples where each tuple contains the start and end times and the speaker name.
    '''
    with open(json_filename, 'r') as f:
        data = json.load(f)
        # expected json structure: {"segments": [
        #                               {
        #                                   "wdlist":[{"start": 0.0, "end": 0.5},...], 
        #                                   "speaker_id": "s1"}, ...
        #                                   ]
        #                            }      
        segments = data['segments']
        times_and_speakers = []
        # epsillon = 0.000001
        for i, segment in enumerate(segments):
            if image_for_each_segment:
                speaker = f"paragraph_{i+1}"
            else:
                speaker = segment['speaker_id'].replace(' ', '_').lower()
            if speakers_to_include is not None and segment['speaker_id'] not in speakers_to_include:
                continue
            wdlist = segment['wdlist']
            prior_end = None
            for wd in wdlist:
                start = wd['start']
                if end_time_seconds is not None and start > end_time_seconds:
                    break
                if prior_end is not None:
                    # adds a segment if there is a gap between the end of the last segment and the start of the current segment
                    # potentially useful for collecting silent speakers for training data
                    # times_and_speakers.append((prior_end+epsillon, start-epsillon, None))
                    pass
                end = wd['end']
                if start_time_seconds is not None and end < start_time_seconds:
                    continue
                prior_end = end
                times_and_speakers.append((start, end, speaker))
    return times_and_speakers
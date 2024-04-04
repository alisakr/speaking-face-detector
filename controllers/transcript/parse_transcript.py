import json

def get_times_and_speakers(
        json_filename=None, 
        transcript_json=None, 
        speakers_to_include=None, 
        start_time_seconds=0, 
        end_time_seconds=None, 
        image_for_each_segment=True,
        include_silent_periods=False,
        speaker_key='speaker_id',
        ignore_first_last_word=False,
        ):
    '''
    Given a json filename, returns a list of tuples where each tuple contains the start and end times and the speaker name.
    '''
    if transcript_json is None and json_filename is None:
        raise ValueError("Either json_filename or transcript_json must be provided")
    if transcript_json is None:
        with open(json_filename, 'r') as f:
            transcript_json = json.load(f)
    # expected json structure: {"segments": [
    #                               {
    #                                   "wdlist":[{"start": 0.0, "end": 0.5, "word": "hi"},...], 
    #                                   speaker_key: "s1"}, ...
    #                                   ]
    #                            }      
    segments = transcript_json['segments']
    times_and_speakers = []
    epsillon = 0.000001
    for i, segment in enumerate(segments):
        if image_for_each_segment:
            speaker = f"paragraph_{i+1}"
        else:
            speaker = segment[speaker_key].replace(' ', '_').lower()
        if speakers_to_include is not None and segment[speaker_key] not in speakers_to_include:
            continue
        wdlist = segment['wdlist']
        prior_end = None
        for i in range(len(wdlist)):
            wd = wdlist[i]
            if ignore_first_last_word and i == 0:
                continue
            if ignore_first_last_word and i == len(wdlist)-1:
                continue
            start = wd['start']
            if end_time_seconds is not None and start > end_time_seconds:
                break
            if include_silent_periods and prior_end is not None:
                # adds a segment if there is a gap between the end of the last segment and the start of the current segment
                # potentially useful for collecting silent speakers for training data
                times_and_speakers.append((prior_end+epsillon, start-epsillon, None))
            end = wd['end']
            if start_time_seconds is not None and end < start_time_seconds:
                continue
            prior_end = end
            times_and_speakers.append((start, end, speaker, wd['word']))
    return times_and_speakers
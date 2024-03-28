import requests

api_key = ""

API_ROOT = "https://app.reduct.video/api/v2/"

def create_project(title, member, api_key_override=api_key):
    proj_data = {
        "title": title,
        # member variable can be org id (available in urls)
        # or email of member
        "member": {member: {"member": True}},
        "team": True,
        "tx_option": "draft"
    }

    res = requests.post(API_ROOT + "create-project",
                        headers={"x-auth-key": api_key_override},
                        json=proj_data)
    if res.status_code != 200:
        raise RuntimeError(f"Failed to create project: {res.text}")
    return res.json()

# res = create_project("Sample Project", "rmo@reduct.video")
# proj_id = res['id']

def create_doc_from_url(proj_id, doc_title, url, api_key_override=api_key):
    doc_data = {"title": doc_title}
    res = requests.post(f"{API_ROOT}project/{proj_id}/create-doc",
                        headers={"x-auth-key": api_key_override},
                        json=doc_data)

    doc_id = res['id']

    res = requests.post(f"{API_ROOT}doc/{doc_id}/import-media",
                        headers={"x-auth-key": api_key_override},
                        json={"url": url}).json()

    return {"doc_id": doc_id, "media_ids": res['media_ids']}

# res = create_doc_from_url(proj_id, "My Youtube Import", "https://youtube.com/...")
# doc_id = res['doc_id']

def create_doc_from_file(proj_id, doc_title, filepath, api_key_override=api_key):
    doc_data = {"title": doc_title}
    res = requests.post(f"{API_ROOT}project/{proj_id}/create-doc",
                        headers={"x-auth-key": api_key_override},
                        json=doc_data)
    if res.status_code != 200:
        raise RuntimeError(f"Failed to create doc: {res.text}")

    doc_id = res.json()['id']

    filename = filepath.split('/')[-1]

    res = requests.post(f"{API_ROOT}doc/{doc_id}/put-media?filename={filename}",
                        headers={"x-auth-key": api_key_override},
                        data=open(filepath, mode='rb'))
    if res.status_code != 200:
        raise RuntimeError(f"Failed to upload media: {res.text}")
    res_json = res.json()
    return {"doc_id": doc_id, "media_id": res_json['media_id']}

def get_transcript_status(doc_id, api_key_override=api_key):
    res = requests.get(f"{API_ROOT}doc/{doc_id}/status",
                        headers={"x-auth-key": api_key_override})
    '''
    Possible statuses: complete...
            {
            "media": {
                "1e3d7641": "transcribed"
            }
        }
    incomplete...
            {
            "media": {
                "e98eecde": "transcribing"
            }
        } 
        and ...
        {
            "media": {
                "e98eecde": "importing"
            }
        }
    '''

    return res.json()

def save_transcript(doc_id, outpath, api_key_override=api_key, format="json"):
    if format not in ("json", "txt", "docx"):
        raise RuntimeError("Invalid transcript format")

    res = requests.get(f"{API_ROOT}doc/{doc_id}/transcript.{format}",
                        headers={"x-auth-key": api_key_override})
    if res.status_code != 200:
        raise RuntimeError(f"Failed to get transcript: {res.text}")
    file = open(outpath, 'wb+')
    result = file.write(res.content)
    file.close()
    return result

def get_transcript(doc_id, api_key_override=api_key, format="json"):
    if format not in ("json", "txt", "docx"):
        raise RuntimeError("Invalid transcript format")

    res = requests.get(f"{API_ROOT}doc/{doc_id}/transcript.{format}",
                        headers={"x-auth-key": api_key_override})
    if res.status_code != 200:
        raise RuntimeError(f"Failed to get transcript: {res.text}")
    return res.json()


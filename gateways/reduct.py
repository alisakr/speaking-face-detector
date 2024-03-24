import requests

api_key = ""

API_ROOT = "https://app.reduct.video/api/v2/"

def create_project(title, member):
    proj_data = {
        "title": title,
        "member": {member: {"member": True}},
        "team": True,
        "tx_option": "draft"
    }

    res = requests.post(API_ROOT + "create-project",
                        headers={"x-auth-key": api_key},
                        json=proj_data)
    return res.json()

# res = create_project("Sample Project", "rmo@reduct.video")
# proj_id = res['id']

def create_doc_from_url(proj_id, doc_title, url):
    doc_data = {"title": doc_title}
    res = requests.post(f"{API_ROOT}project/{proj_id}/create-doc",
                        headers={"x-auth-key": api_key},
                        json=doc_data)

    doc_id = res['id']

    res = requests.post(f"{API_ROOT}doc/{doc_id}/import-media",
                        headers={"x-auth-key": api_key},
                        json={"url": url}).json()

    return {"doc_id": doc_id, "media_ids": res['media_ids']}

# res = create_doc_from_url(proj_id, "My Youtube Import", "https://youtube.com/...")
# doc_id = res['doc_id']

def create_doc_from_file(proj_id, doc_title, filepath):
    doc_data = {"title": doc_title}
    res = requests.post(f"{API_ROOT}project/{proj_id}/create-doc",
                        headers={"x-auth-key": api_key},
                        json=doc_data)

    doc_id = res['id']

    filename = filepath.split('/')[-1]

    res = requests.post(f"{API_ROOT}doc/{doc_id}/put-media?filename={filename}",
                        headers={"x-auth-key": api_key},
                        data=open(filepath, mode='rb')).json()

    return {"doc_id": doc_id, "media_id": res['media_id']}

def get_transcript_status(doc_id):
    res = requests.get(f"{API_ROOT}doc/{doc_id}/status",
                        headers={"x-auth-key": api_key})

    return res.json()

def save_transcript(doc_id, outpath, api_key_override=api_key, format="json"):
    if format not in ("json", "txt", "docx"):
        raise RuntimeError("Invalid transcript format")

    res = requests.get(f"{API_ROOT}doc/{doc_id}/transcript.{format}",
                        headers={"x-auth-key": api_key_override})
    file = open(outpath, 'wb+')
    result = file.write(res.content)
    file.close()
    return result
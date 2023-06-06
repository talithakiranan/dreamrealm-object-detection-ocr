import requests

def download_model(id, destination):
    url = "https://drive.google.com/uc?id=12EqvdV9Fg3M7Q5AtbzzMoFb_6IoMr1bN&export=download"
    
    session = requests.Session()

    response = session.get(url, params={'id':id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id':id, 'confirm':token}
        response = session.get(url, params = params, steam = True)
    
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value 
        
    return None 

def save_response_content(response, destination):
    CHUNK = 32768

    with open(destination, 'wb') as file:
        for chunk in response.iter_content(CHUNK):
            if chunk: #filter out keep-alive new chunkc
                file.write(chunk)

    response.raise_for_status()

    with open(destination, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
import requests

with open(r'C:\Users\Masoa\OneDrive\work\CTAI\CTAI_flask\uploads\10007.dcm', 'rb') as f:
    r = requests.post('http://127.0.0.1:5003/upload', 
                     files={'file': ('10007.dcm', f, 'application/dicom')})
    print('STATUS:', r.status_code)
    print('RESPONSE:', r.json())

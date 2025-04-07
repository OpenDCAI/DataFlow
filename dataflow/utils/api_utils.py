from openai import OpenAI
import json
import requests
import os

def api_chat( 
            system_info: str,
            messages: str,
            model: str,
            api_url : str = "",
            api_key : str = "",
            finish_try: int = 3,
            mode_test : bool = True
            ):
    if api_key == "":
        api_key = os.environ.get("API_KEY")

    if api_key is None:
        raise ValueError("Lack of API_KEY")
    
    if mode_test is True:
        try:
            payload = json.dumps({
                "model": model,
                "messages": [
                    {"role": "system", "content": system_info},
                    {"role": "user", "content": messages}
                ]
            })

            headers = {
            'Authorization': f"Bearer {api_key}",
            'Content-Type': 'application/json',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)'
            }
            
            # request OpenAI API
            response = requests.post(api_url, headers=headers, data=payload, timeout=1800)
            
            # API debug code
            # print("response ", response)
            # print("response.status_code", response.status_code)

            if response.status_code == 200:
                response_data = response.json()
                return response_data['choices'][0]['message']['content']


        except Exception as e:
            print("Error:", e)
            pass
    
    else :
        client = OpenAI(api_key=api_key)
        api_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_info},
                {"role": "user", "content": messages}
            ]
        )

        response_content = api_response.choices[0].message.content.strip()
        
        return response_content
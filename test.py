# test.py

from backend import classify_message

user_input = "[배송조회] 9/9 고객주소가 잘못되었습니다. 택배가 반송되었습니다. 배송 주소 수정 uuuu.me/FgMRD7"
result = classify_message(user_input)

import json
print(json.dumps(result, indent=2, ensure_ascii=False))
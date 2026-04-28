import requests

url = "https://default6a7edf98d34f4386aa41a9042c7189.ec.environment.api.powerplatform.com:443/powerautomate/automations/direct/workflows/26f565bb4e584f649cc90342649ae618/triggers/manual/paths/invoke?api-version=1&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=nieOmMvo9Av2iAj03xaMCNB_I7bxA2dNPJrw-X5u0RE"

payload = {
    "start_time": "2026-03-23 17:30:00+09:00",
    "person_id": 2,
    "frames_total": 46,
    "state": "NG_helmet&harness",
    "xyxy": [135.0, 190.0, 623.0, 479.0]
}

r = requests.post(url, json=payload)
print(r.status_code)

"""🚨 AI SAFETY ALERT

⏰ Time: @{triggerBody()?['start_time']}
👤 Person ID: @{triggerBody()?['person_id']}
🎞 Frames: @{triggerBody()?['frames_total']}

❌ Violation: @{triggerBody()?['state']}

📦 Bounding box:
@{triggerBody()?['xyxy']}"""
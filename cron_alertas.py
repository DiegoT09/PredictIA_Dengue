import requests
from datetime import date

semana = date.today().isocalendar()[1]
anio   = date.today().year

url = f"https://predictia-dengue.onrender.com/alertas/guardar?semana={semana}&anio={anio}"
resp = requests.get(url, timeout=120)
print(resp.json())


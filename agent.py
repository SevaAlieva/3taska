import os
import pandas as pd
import numpy as np
import json
import base64
from openai import OpenAI
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
def safe_execute(code, df):
    dangerous = ['os.system', 'subprocess', '__import__', 'eval(', 'exec(', 'open(', 'rm ', 'del ', 'os.remove']
    for bad in dangerous:
        if bad in code.lower():
            return None, f"обнаружен опасный код: {bad}"

    os.makedirs("charts", exist_ok=True)
    local_vars = {
        "df": df,
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns
    }
    try:
        exec(code, local_vars)
        charts = []
        for file in os.listdir("charts"):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                with open(os.path.join("charts", file), "rb") as f:
                    charts.append({
                        "name": file,
                        "data": base64.b64encode(f.read()).decode()
                    })
                os.remove(os.path.join("charts", file))
        plt.close('all')
        return charts, None
    except Exception as e:
        return None, str(e)


def analyze_data(df, user_text):
    if len(user_text) > 2000:
        return {"report": "слишком длинный запрос", "charts": None, "ok": False}
    dangerous_words = [
        "игнорируй", "ignore", "забудь", "forget", "отключи", "обойди",
        "удали файл", "rm -rf", "delete", "subprocess", "твоя роль",
        "скачай", "отправь ключ", "send key", "отключи ограничения",
        "выполни код", "os.system"
    ]
    for word in dangerous_words:
        if word in user_text.lower():
            return {"report": f"обнаружено запрещенное слово: {word}", "charts": None, "ok": False}
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    data_info = f"""
- количество строк: {df.shape[0]}
- количество колонок: {df.shape[1]}
- названия колонок: {list(df.columns)}
- типы данных: {df.dtypes.to_dict()}
- пропуски по колонкам: {df.isnull().sum().to_dict()}
- числовые колонки: {numeric_cols}
- категориальные колонки: {categorical_cols[:10]}
{df.describe().to_string() if len(numeric_cols) > 0 else "нет числовых колонок"}
"""
    prompt = f"""
Ты профессиональный AI-аналитик данных. Тебе предоставлен pandas DataFrame.
{data_info}
{'''дополнительный запрос пользователя:''' + user_text if user_text else '''нет дополнительного запроса. 
Проведи полный анализ данных самостоятельно'''}

Обязательная структура твоего ответа: если нет запроса пользователя,
ты должен провести полный EDA анализ в следующем формате:

1. общее описание датасета:
- размер и структура
- описание признаков
- типы данных

2. качество данных:
- пропуски и как с ними быть
- дубликаты
- аномалии и выбросы

3. основные статистики:
- средние, медианы, стандартные отклонения
- минимумы и максимумы
- распределения значений

4. корреляции и зависимости:
- сильные корреляции (если есть числовые колонки)
- взаимосвязи между признаками

5. ответ на запрос пользователя

6. итоговое выводы:
- ключевые выводы
- рекомендации

Ты обязан сначала провести полный EDA-анализ датасета, а только потом
отдельно ответить на дополнительный запрос пользователя.

Правила создания графиков:
1. создавай только действительно полезные графики (МАКСИМУМ ДО 4!)
2. сохраняй графики в папку "charts/"
3. используй осмысленные имена файлов
4. разрешенные графики:
   - тепловая карта корреляций (heatmap)
   - гистограммы распределений (hist)
   - boxplot для выбросов
   - scatter plot для зависимостей
5. для корреляций всегда используй df[numeric_cols].corr()
6. для категориальных колонок не строй гистограммы

Требования к коду:
- импортируй: import matplotlib.pyplot as plt, import seaborn as sns
- каждый график должен сопровождаться print() с пояснением
- сохраняй графики: plt.savefig('charts/название.png')
- закрывай фигуры: plt.close()

Формат твоего ответа:
Ты должен вернуть ТОЛЬКО JSON в формате:
{{"code": "твой код", "report": "текстовый отчет"}}
"""

    try:
        response = client.chat.completions.create(
            model= "nvidia/nemotron-3-super-120b-a12b:free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=4000
        )
        content = response.choices[0].message.content
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0]
        elif '```' in content:
            content = content.split('```')[1].split('```')[0]
        content = content.strip()
        if not content.startswith('{'):
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                content = content[start:end]
        result = json.loads(content)
        code = result.get("code", "")
        report = result.get("report", "")

        if not report:
            report = "анализ выполнен успешно"
        charts, err = safe_execute(code, df)
        if err:
            return {"report": f"ошибка выполнения: {err}", "charts": None, "ok": False}
        return {"report": report, "charts": charts, "ok": True}
    except json.JSONDecodeError as e:
        return {"report": f"ошибка формата JSON: {str(e)}", "charts": None, "ok": False}
    except Exception as e:
        return {"report": f"ошибка: {str(e)}", "charts": None, "ok": False}
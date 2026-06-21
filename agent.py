import os
import pandas as pd
import numpy as np
import json
import base64
from openai import OpenAI
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import re

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

    all_charts = []
    final_report = ""

    eda_prompt = f"""
Ты профессиональный AI-аналитик данных. Тебе предоставлен pandas DataFrame.
{data_info}

Проведи первичный EDA-анализ данных. Определи:
1. Основные характеристики датасета
2. Качество данных (пропуски, дубликаты, выбросы)
3. Базовые статистики
4. Какие графики нужно построить для визуализации (максимум 2-3)

Верни ТОЛЬКО JSON в формате:
{{"code": "код для графиков и анализа", "report": "краткий отчет по EDA"}}

Обязательно импортируй matplotlib и seaborn в коде.
Сохраняй графики в папку "charts/".
"""
    try:
        response = client.chat.completions.create(
            model="nvidia/nemotron-3-super-120b-a12b:free",
            messages=[{"role": "user", "content": eda_prompt}],
            temperature=0.1,
            max_tokens=2000
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

        charts, err = safe_execute(code, df)
        if err:
            return {"report": f"ошибка выполнения EDA: {err}", "charts": None, "ok": False}

        if charts:
            all_charts.extend(charts)
        final_report += f"## Шаг 1: Первичный EDA-анализ\n\n{report}\n\n"

        if user_text and user_text.strip():
            detailed_prompt = f"""
Ты профессиональный AI-аналитик данных. 
Результаты первичного EDA-анализа:
{report}

Теперь ответь на дополнительный запрос пользователя:
{user_text}

Проведи детальный анализ для ответа на этот запрос. Если нужно, построй дополнительные графики.
Верни ТОЛЬКО JSON в формате:
{{"code": "код для дополнительных графиков", "report": "детальный ответ на запрос"}}

Сохраняй графики в папку "charts/" с новыми именами.
"""
            response2 = client.chat.completions.create(
                model="nvidia/nemotron-3-super-120b-a12b:free",
                messages=[{"role": "user", "content": detailed_prompt}],
                temperature=0.1,
                max_tokens=2000
            )

            content2 = response2.choices[0].message.content
            if '```json' in content2:
                content2 = content2.split('```json')[1].split('```')[0]
            elif '```' in content2:
                content2 = content2.split('```')[1].split('```')[0]
            content2 = content2.strip()
            if not content2.startswith('{'):
                start = content2.find('{')
                end = content2.rfind('}') + 1
                if start != -1 and end > start:
                    content2 = content2[start:end]

            result2 = json.loads(content2)
            code2 = result2.get("code", "")
            report2 = result2.get("report", "")

            charts2, err2 = safe_execute(code2, df)
            if err2:
                return {"report": f"ошибка выполнения детального анализа: {err2}", "charts": None, "ok": False}

            if charts2:
                all_charts.extend(charts2)
            final_report += f"## Шаг 2: Детальный анализ по запросу\n\n{report2}\n\n"

        if user_text and user_text.strip():
            conclusion_prompt = f"""
На основе проведенного анализа:
EDA-анализ: {report}
Ответ на запрос пользователя ({user_text}): {report2}

Сформулируй итоговые выводы и рекомендации по данным.
Ответ должен быть кратким и практичным.
"""
        else:
            conclusion_prompt = f"""
На основе проведенного EDA-анализа:
{report}

Сформулируй итоговые выводы и рекомендации по данным.
Ответ должен быть кратким и практичным.
"""

        response3 = client.chat.completions.create(
            model="nvidia/nemotron-3-super-120b-a12b:free",
            messages=[{"role": "user", "content": conclusion_prompt}],
            temperature=0.1,
            max_tokens=1000
        )

        final_report += f"## Итоговые выводы\n\n{response3.choices[0].message.content}"

        return {"report": final_report, "charts": all_charts, "ok": True}

    except json.JSONDecodeError as e:
        return {"report": f"ошибка формата JSON: {str(e)}", "charts": None, "ok": False}
    except Exception as e:
        return {"report": f"ошибка: {str(e)}", "charts": None, "ok": False}
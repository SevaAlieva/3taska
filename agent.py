import os
import pandas as pd
import numpy as np
import base64
import re
import io
import sys
from openai import OpenAI
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()

def safe_execute(code, df):
    dangerous = ['os.system', 'subprocess', '__import__', 'eval(', 'exec(', 'open(', 'rm ', 'del ', 'os.remove']
    for bad in dangerous:
        if bad in code.lower():
            return None, f"обнаружен опасный код: {bad}"
    code = re.sub(r'^(\s*)(import |from )', r'\1# ', code, flags=re.MULTILINE)
    local_vars = {
        "df": df.copy() if df is not None else None,
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns
    }
    old_out = sys.stdout
    sys.stdout = buf = io.StringIO()

    try:
        exec(code, local_vars)
        output = buf.getvalue()
        plot_data = None

        if plt.gcf().get_axes():
            bio = io.BytesIO()
            plt.savefig(bio, format='png', bbox_inches='tight', dpi=100)
            bio.seek(0)
            plot_data = base64.b64encode(bio.read()).decode()
            plt.close()
        plt.close('all')
        return {"output": output, "plot_data": plot_data}, None
    except Exception as e:
        return None, str(e)
    finally:
        sys.stdout = old_out

def call_llm(messages, api_key):
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    response = client.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        messages=messages,
        temperature=0.2,
        max_tokens=4000
    )
    return response.choices[0].message.content

def extract_code(text):
    pattern = r'```python\s*(.*?)```'
    match = re.search(pattern, text, re.DOTALL | re.I)
    return match.group(1).strip() if match else None

def analyze_data(df, user_text, api_key):
    if user_text and len(user_text) > 2000:
        return {"report": "слишком длинный запрос", "charts": None, "ok": False}

    dangerous_words = [
        "игнорируй", "ignore", "забудь", "forget", "отключи", "обойди",
        "удали файл", "rm -rf", "delete", "subprocess", "твоя роль",
        "скачай", "отправь ключ", "send key", "отключи ограничения",
        "os.system"
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

    system_prompt = f"""Ты — агент-аналитик данных. У тебя есть доступ к pandas DataFrame (переменная `df`).
Информация о данных:
{data_info}
Ты можешь выполнять Python-код для анализа данных. Оберни код в блок ```python ... ```.
После выполнения кода ты получишь результат и сможешь:
1. Написать новый код для продолжения анализа
2. Сформулировать финальный отчёт на основе полученных данных

КРИТИЧЕСКИ ВАЖНО:
- Делай выводы ТОЛЬКО на основе РЕАЛЬНЫХ данных из выполнения кода
- НЕ пиши "вероятно", "можно предположить", "ожидается"
- Используй КОНКРЕТНЫЕ ЧИСЛА: средние значения, корреляции, проценты
- Если ты не выполнил код для получения данных — не делай выводов об этих данных
- Отвечай ИСКЛЮЧИТЕЛЬНО на русском языке

Правила:
- Используй print() для вывода результатов
- Для графиков используй plt.plot(), plt.bar(), plt.hist(), sns.heatmap() и т.д.
- Не используй plt.show() - графики сохраняются автоматически
- Не импортируй os, sys, subprocess и другие системные модули

Работай итеративно: выполняй код, анализируй результаты, при необходимости выполняй ещё код.
Когда у тебя будет достаточно информации для полного отчёта — просто напиши финальный отчёт БЕЗ блока кода."""

    task = user_text if user_text and user_text.strip() else "Проведи полный EDA-анализ: статистика, визуализация, корреляции, инсайты."
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": f"Задача: {task}\n\nНачни анализ. Вызывай код через ```python ... ```, получай результаты и на их основе формируй выводы."}
    ]
    all_charts = []
    final_report = ""
    max_steps = 5
    for step in range(max_steps):
        try:
            llm_response = call_llm(conversation, api_key)
            conversation.append({"role": "assistant", "content": llm_response})
            if len(conversation) > 9:
                conversation = [conversation[0]] + conversation[-8:]
            code = extract_code(llm_response)

            if code:
                exec_result, err = safe_execute(code, df)
                observation = ""
                if err:
                    observation = f"Ошибка выполнения кода: {err}"
                else:
                    observation = "Код успешно выполнен.\n"
                    if exec_result["output"]:
                        output_text = exec_result['output'][:2000]
                        if len(exec_result['output']) > 2000:
                            output_text += "\n... (вывод обрезан)"
                        observation += f"Вывод:\n{output_text}\n"
                    if exec_result["plot_data"]:
                        all_charts.append({
                            "name": f"chart_{len(all_charts) + 1}.png",
                            "data": exec_result["plot_data"]
                        })
                        observation += f"График построен и сохранён (всего графиков: {len(all_charts)}).\n"
                conversation.append({
                    "role": "user",
                    "content": f"Результат выполнения кода (шаг {step + 1}):\n{observation}\n\nПродолжай анализ или сформулируй финальный отчёт."
                })
            else:
                final_report = llm_response
                break
        except Exception as e:
            final_report = f"Ошибка в агентном цикле: {e}"
            break
    if not final_report:
        conversation.append({
            "role": "user",
            "content": "Анализ завершён. Сформулируй итоговый отчёт на основе всех полученных данных. НЕ пиши больше код, только текстовый отчёт с выводами и инсайтами."
        })
        try:
            final_report = call_llm(conversation, api_key)
        except Exception as e:
            final_report = f"Анализ завершён, но не удалось сформировать отчёт: {e}"
    return {"report": final_report, "charts": all_charts, "ok": True}
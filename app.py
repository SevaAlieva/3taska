import os
import streamlit as st
import pandas as pd
import base64
from io import BytesIO
from agent import analyze_data

blocked_prompts = [
    'забудь предыдущие инструкции', 'игнорируй предыдущие инструкции', 'отключи ограничения',
    'обойди ограничения', 'выполни код', 'удали файл', 'удали папку', 'скачай файл',
    'отправь данные', 'отправь файл', 'твоя роль', 'os.system', 'subprocess',
    'eval(', 'exec(', 'отключи безопасность', 'игнорируй правила', 'отправь ключ']

def is_prompt_safe(prompt):
    if not prompt:
        return True
    prompt_lower = prompt.lower()
    for pattern in blocked_prompts:
        if pattern.lower() in prompt_lower:
            return False
    return True

st.set_page_config(page_title="Задание 3: аналитик данных", layout="wide")
st.title("Анализ данных через нейросеть")

st.markdown("""
### Инструкция:
1. Загрузите файл с данными (CSV или Excel)
2. Напишите, что нужно проанализировать (или оставьте пустым для полного анализа)
3. Нажмите "Запустить анализ"
""")

uploaded = st.file_uploader("Выберите файл", type=["csv", "xlsx", "xls"])

example = """Пример запроса: найди корреляции между колонками и построй тепловую карту"""

user_q = st.text_area("Ваш запрос",
                      placeholder=example, height=100)
if uploaded:
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        elif uploaded.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded)
        else:
            st.error("неподдерживаемый формат. Используйте CSV или Excel")
            st.stop()
    except Exception as e:
        st.error(f"ошибка при загрузке файла: {e}")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("строк", df.shape[0])
    with col2:
        st.metric("колонок", df.shape[1])
    with col3:
        st.metric("числовых колонок", len(df.select_dtypes(include=['number']).columns))
    with col4:
        st.metric("пропуски", df.isnull().sum().sum())

    with st.expander("предпросмотр данных"):
        st.subheader("первые 5 строк")
        st.dataframe(df.head())

        st.subheader("информация о колонках")
        col_info = pd.DataFrame({
            'тип': df.dtypes,
            'пропуски': df.isnull().sum(),
            'уникальные': df.nunique()
        })
        st.dataframe(col_info)

    if st.button("Запустить анализ", type="primary"):
        # Проверка безопасности промпта
        if not is_prompt_safe(user_q):
            st.error("обнаружен небезопасный запрос, пожалуйста, переформулируйте вопрос.")
            st.stop()
        for file in os.listdir("charts"):
            os.remove(os.path.join("charts", file))

        with st.spinner("нейросеть анализирует данные..."):
            try:
                result = analyze_data(df, user_q)
            except Exception as e:
                st.error(f"ошибка при анализе: {e}")
                st.stop()

        if result["ok"]:
            st.success("анализ успешно завершен!")
            with st.expander("отчет анализа", expanded=True):
                st.markdown(result["report"])

            if result["charts"]:
                st.subheader("сгенерированные графики")
                cols = st.columns(2)
                for idx, chart in enumerate(result["charts"]):
                    with cols[idx % 2]:
                        st.image(BytesIO(base64.b64decode(chart["data"])),
                                 caption=chart["name"], use_column_width=True)
            else:
                st.info("графики не были сгенерированы(возможно, нет подходящих данных)")
        else:
            st.error(f"{result['report']}")
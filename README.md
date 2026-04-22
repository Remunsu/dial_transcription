# Требования 
Windows 10\
ffmpeg в переменной среды PATH\
Python не старше 3.10

# Установка
Открыть термнал в папке с файлом и выполнить следующие команды:\
python -m venv venv\
source venv/Scripts/activate\
pip install -r requirements.txt\
hf download ai-sage/GigaAM-v3 --local-dir ./gigaam-v3\
hf download neurlang/ipa-whisper-medium --local-dir ./ipa-whisper-medium

# Инструкция
1) Запустить ST3.py
2) Выбрать аудиофайл
3) Нажать "Начать распознавание"

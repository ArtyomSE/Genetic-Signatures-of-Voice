# Genetic-Signatures-of-Voice

## Сборка Docker-образа
```
docker build -t word_approach:local .
```

## Запуск образа 
```
docker run --gpus all --rm -v $(pwd):/app word_approach:local python3 main.py input_path output_path diarization/ 0.5 40

* input_path - путь к входным данным

* output_path - путь к папке, в которой будут результаты

**Важно**, чтобы в конце input_path и output_path был символ '/'

Пример запуска:
docker run --gpus all --rm -v $(pwd):/app word_approach:local python3 main.py wavs/ word_approach/ diarization/ 0.5 40
```

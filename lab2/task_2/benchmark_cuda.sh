#!/bin/bash

PROGRAM="./task2_cuda"

T_END=5.0
RUNS=10

if [ ! -f "$PROGRAM" ]; then
    echo "Ошибка: Программа $PROGRAM не найдена."
    exit 1
fi

echo "=========================================================="
echo " ЗАПУСК ТЕСТОВ ПРОИЗВОДИТЕЛЬНОСТИ (CUDA)"
echo " Усреднение по $RUNS запускам для каждого размера."
echo "=========================================================="

printf "%-10s | %-15s\n" "N (Тел)" "Avg Time (sec)"
echo "------------------------------"

for N in 16 64 128 256 512 1024 2048 4096 8192; do
    INPUT_FILE="input_$N.txt"

    if [ ! -f "$INPUT_FILE" ]; then
        echo "Файл $INPUT_FILE не найден, пропускаем..."
        continue
    fi

    total_time=0

    for ((i=1; i<=RUNS; i++)); do
        time_val=$($PROGRAM $T_END $INPUT_FILE)

        total_time=$(echo "$total_time + $time_val" | bc -l)
    done

    # Считаем среднее
    avg_time=$(echo "scale=6; $total_time / $RUNS" | bc -l)

    # Выводим строку таблицы
    printf "%-10d | %-15s\n" "$N" "$avg_time"
done

echo "=========================================================="
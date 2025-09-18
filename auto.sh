#!/bin/bash

# --- 配置 ---
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# 清空所有旧日志文件
echo "清空旧日志文件..."
find "$LOG_DIR" -name "*.log" -type f -exec truncate -s 0 {} \;

# --- 启动 Learner ---
echo "正在启动 Learner (GPU 2)..."
nohup bash -c "CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false python -u learner.py" \
    >> "$LOG_DIR/2_gpu1.log" 2>&1 &
LEARNER_PID=$!
echo "启动: learner.py (GPU 2) - 日志: $LOG_DIR/2_gpu1.log [PID: $LEARNER_PID]"

# --- 启动 Actors (GPU 0) ---
echo "正在启动 Actors (GPU 0)..."
ACTOR_WATCHER_PIDS=() # 存储所有 actor 守护进程的 PID
ACTOR_PIDS_FILE="$LOG_DIR/active_actor_pids.txt" # 用于存储当前活跃 actor 的 PID
> "$ACTOR_PIDS_FILE" # 清空 PID 文件

for i in {1..8}; do
    # 为每个 actor 创建一个独立的守护进程
    (
        while true; do
            echo "[$(date)] 启动 Actor ${i} (GPU 0)..."
            # 启动 actor 并记录其 PID
            CUDA_VISIBLE_DEVICES=3 python -u actor.py >> "$LOG_DIR/1_gpu0_${i}.log" 2>&1 &
            ACTOR_PID=$!
            echo $ACTOR_PID >> "$ACTOR_PIDS_FILE"
            echo "[$(date)] Actor ${i} (GPU 0) 已启动 [PID: $ACTOR_PID]"

            # 等待 actor 进程结束
            # wait $ACTOR_PID
            # EXIT_CODE=$?
            # echo "[$(date)] Actor ${i} (GPU 0) 退出，退出码: $EXIT_CODE"
            # ========== 核心替换：使用 pgrep 轮询 ==========
            # 循环检查进程是否还活着
            while pgrep -P $$ > /dev/null 2>&1; do
                # 检查我们启动的这个特定 PID 是否还在进程列表中
                if kill -0 $ACTOR_PID 2>/dev/null; then
                    sleep 1 
                else
                    echo "[$(date)] DEBUG: PID $ACTOR_PID is NOT alive. Breaking loop."
                    break # 进程已退出，跳出循环
                fi
            done
            echo "[$(date)] Actor ${i} (GPU 0) 进程 (PID: $ACTOR_PID) 已退出。"

            # 从活跃 PID 列表中移除该 PID
            sed -i "/^$ACTOR_PID\$/d" "$ACTOR_PIDS_FILE"

            echo "[$(date)] Actor ${i} 将在5秒后重新启动..."
            sleep 5
        done
    ) &
    ACTOR_WATCHER_PIDS+=($!)
    echo "Actor ${i} (GPU 0) 守护进程已启动 [PID: ${ACTOR_WATCHER_PIDS[-1]}]"
done

# --- 启动 Actors (GPU 1) ---
echo "正在启动 Actors (GPU 1)..."
for i in {1..8}; do
    # 为每个 actor 创建一个独立的守护进程
    (
        while true; do
            echo "[$(date)] 启动 Actor ${i} (GPU 1)..."
            CUDA_VISIBLE_DEVICES=1 python -u actor.py >> "$LOG_DIR/1_gpu1_${i}.log" 2>&1 &
            ACTOR_PID=$!
            echo $ACTOR_PID >> "$ACTOR_PIDS_FILE"
            echo "[$(date)] Actor ${i} (GPU 1) 已启动 [PID: $ACTOR_PID]"

            # 等待 actor 进程结束
            # wait $ACTOR_PID
            # EXIT_CODE=$?
            # echo "[$(date)] Actor ${i} (GPU 1) 退出，退出码: $EXIT_CODE"
            # ========== 核心替换：使用 pgrep 轮询 ==========
            # 循环检查进程是否还活着
            while pgrep -P $$ > /dev/null 2>&1; do
                # 检查我们启动的这个特定 PID 是否还在进程列表中
                if kill -0 $ACTOR_PID 2>/dev/null; then
                    sleep 1 
                else
                    break # 进程已退出，跳出循环
                fi
            done
            echo "[$(date)] Actor ${i} (GPU 1) 进程 (PID: $ACTOR_PID) 已退出。"
            # 从活跃 PID 列表中移除该 PID
            sed -i "/^$ACTOR_PID\$/d" "$ACTOR_PIDS_FILE"

            echo "[$(date)] Actor ${i} 将在5秒后重新启动..."
            sleep 5
        done
    ) &
    ACTOR_WATCHER_PIDS+=($!)
    echo "Actor ${i} (GPU 1) 守护进程已启动 [PID: ${ACTOR_WATCHER_PIDS[-1]}]"
done

# --- 一级守护：监控 Learner ---
echo "所有进程已启动。进入守护模式，监控 Learner 进程 (PID: $LEARNER_PID)..."

# 等待 Learner 进程结束
wait $LEARNER_PID
LEARNER_EXIT_CODE=$?

echo "Learner 进程 (PID: $LEARNER_PID) 已退出，退出码: $LEARNER_EXIT_CODE"

# Learner 退出后，执行级联终止
echo "正在终止所有 Actor 守护进程和活跃的 Actor 进程..."

# 1. 先杀死所有守护进程（防止它们拉起新的 actor）
for pid in "${ACTOR_WATCHER_PIDS[@]}"; do
    if kill -0 $pid 2>/dev/null; then
        echo "正在终止 Actor 守护进程 (PID: $pid)..."
        kill $pid
        wait $pid 2>/dev/null # 等待守护进程退出
    fi
done

# 2. 再杀死所有当前活跃的 actor 进程
if [ -f "$ACTOR_PIDS_FILE" ]; then
    while IFS= read -r pid; do
        if [[ -n "$pid" ]] && kill -0 $pid 2>/dev/null; then
            echo "正在终止活跃的 Actor 进程 (PID: $pid)..."
            kill $pid
            # 等待进程优雅退出，最多5秒
            for _ in {1..5}; do
                if ! kill -0 $pid 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            # 如果5秒后还未退出，强制杀死
            if kill -0 $pid 2>/dev/null; then
                echo "活跃的 Actor (PID: $pid) 未响应，强制终止..."
                kill -9 $pid
            fi
        fi
    done < "$ACTOR_PIDS_FILE"
fi

echo "所有进程已终止。训练任务结束。"
exit $LEARNER_EXIT_CODE


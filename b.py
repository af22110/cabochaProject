import psutil
import os

def get_available_memory_gb():
    """
    システム全体で利用可能なメモリ容量をGB単位で返す。
    """
    virtual_mem = psutil.virtual_memory()
    available_gb = virtual_mem.available / (1024 ** 3) # バイトからGBに変換
    return available_gb

def get_total_memory_gb():
    """
    システム全体の総メモリ容量をGB単位で返す。
    """
    virtual_mem = psutil.virtual_memory()
    total_gb = virtual_mem.total / (1024 ** 3)
    return total_gb

def get_memory_usage_by_current_process_gb():
    """
    現在のPythonプロセスが使用しているメモリ量をGB単位で返す。
    (RSS: Resident Set Size - 物理メモリ上のサイズ)
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    rss_gb = memory_info.rss / (1024 ** 3)
    return rss_gb

if __name__ == "__main__":
    total_mem_gb = get_total_memory_gb()
    available_mem_gb = get_available_memory_gb()
    current_process_mem_gb = get_memory_usage_by_current_process_gb()

    print(f"システム全体の総メモリ容量: {total_mem_gb:.2f} GB")
    print(f"現在利用可能なメモリ容量: {available_mem_gb:.2f} GB")
    print(f"このPythonプロセスが使用中のメモリ容量: {current_process_mem_gb:.2f} GB")

    # N-gram読み込みなどの重い処理の前に呼び出す
    # print(f"\n処理開始前の利用可能メモリ: {get_available_memory_gb():.2f} GB")
    # ... (N-gram読み込み処理) ...
    # print(f"処理終了後の利用可能メモリ: {get_available_memory_gb():.2f} GB")
    # print(f"処理終了後のプロセス使用メモリ: {get_memory_usage_by_current_process_gb():.2f} GB")
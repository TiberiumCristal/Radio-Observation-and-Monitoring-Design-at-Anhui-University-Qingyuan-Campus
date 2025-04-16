import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import psutil
import gc
import tempfile
import shutil

def batch_iq_analysis(
    input_dir: str,
    output_dir: str = "spectrum_results",
    fft_size: int = 8192,
    overlap: int = 4096, #分段大小，最大不超过fft_size的一半
    window_type: str = 'blackman', #加窗函数选择
    center_freq: float = 1420.40575177e6,
    dpi: int = 150,
    max_memory_usage: float = 80.0, #最大内存使用百分比
    enable_diff: bool = False #是否开启相邻差分
):
    
    
    # 创建输出目录和临时目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # 获取并排序文件列表
        input_path = Path(input_dir)
        files = sorted(input_path.glob("*.wav"), key=lambda f: f.name)
        if not files:
            raise FileNotFoundError(f"输入目录中未找到WAV文件: {input_dir}")
        print(f"发现 {len(files)} 个待处理文件")

        # 内存监控初始化
        mem = psutil.virtual_memory()
        safe_memory = mem.total * (max_memory_usage / 100)

        # 存储PSD数据用于差分计算
        psd_cache = []

        # 顺序处理主循环
        with tqdm(total=len(files), desc="处理进度", unit="file") as pbar:
            for idx, file in enumerate(files):
                try:
                    # 内存检查
                    if psutil.virtual_memory().used > safe_memory:
                        raise MemoryError(f"内存使用超过{max_memory_usage}%阈值")

                    # 处理单个文件
                    psd_data = process_file(
                        file, output_path, temp_dir,
                        fft_size, overlap, window_type.lower(),
                        center_freq, dpi, enable_diff
                    )
                    
                    # 缓存PSD数据用于差分
                    if enable_diff:
                        psd_cache.append((file.stem, psd_data))

                    # 主动释放资源
                    gc.collect()
                    plt.close('all')

                except Exception as e:
                    print(f"\n文件处理失败 {file.name}: {str(e)}")
                finally:
                    pbar.update(1)

            # 执行差分处理
            if enable_diff and len(psd_cache) > 1:
                process_difference(output_path, psd_cache, fft_size, center_freq, dpi)

    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"处理完成 | 结果保存至: {output_path.absolute()}")

def process_file(
    file_path: Path,
    output_dir: Path,
    temp_dir: Path,
    fft_size: int,
    overlap: int,
    window_type: str,
    center_freq: float,
    dpi: int,
    enable_diff: bool
) -> tuple:
    """处理单个文件并返回PSD数据"""
    try:
        # 读取数据
        fs, data = wavfile.read(file_path)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("必须为双通道I/Q格式")

        # 转换为复数信号
        iq_signal = (data[:, 0].astype(np.float32) + 1j * data[:, 1].astype(np.float32)) / 32767.0
        del data

        # 参数验证
        if (fft_size & (fft_size-1)) != 0:
            raise ValueError("FFT长度必须是2的幂")
        if overlap >= fft_size:
            raise ValueError("重叠量不能超过FFT长度")

        # 窗函数配置
        window_dict = {
            'hann': np.hanning,
            'hamming': np.hamming,
            'blackman': np.blackman,
            'flattop': lambda n: np.kaiser(n, 8.6)
        }
        window = window_dict[window_type](fft_size)

        # 分块频谱计算
        step = fft_size - overlap
        num_frames = (len(iq_signal) - fft_size) // step + 1
        psd = np.zeros(fft_size, dtype=np.float32)

        for i in range(num_frames):
            start = i * step
            segment = iq_signal[start:start+fft_size] * window
            spectrum = np.fft.fftshift(np.fft.fft(segment))
            psd += np.abs(spectrum) ** 2
            del segment, spectrum

        # 功率谱计算
        psd /= (num_frames * np.sum(window**2))
        psd_db = 10 * np.log10(psd + 1e-20)
        freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, 1/fs)) + center_freq

        # 保存PSD数据到临时文件
        if enable_diff:
            temp_file = temp_dir / f"{file_path.stem}.npy"
            np.save(temp_file, (freqs, psd_db))

        # 生成标准频谱图
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(freqs/1e6, psd_db, linewidth=0.6)
        ax.set_title(f"{file_path.stem}\nRBW: {fs/fft_size:.1f} Hz", fontsize=10)
        ax.set_xlabel("Frequency [MHz]", fontsize=9)
        ax.set_ylabel("PSD [dB/Hz]", fontsize=9)
        ax.set_xlim((center_freq - fs/2)/1e6, (center_freq + fs/2)/1e6)
        ax.grid(True, which='both', alpha=0.3, linestyle='--')
        plt.savefig(output_dir / f"{file_path.stem}_spectrum.png", dpi=dpi, bbox_inches='tight')
        plt.close(fig)

        return (freqs, psd_db)

    except Exception as e:
        raise RuntimeError(f"处理失败: {str(e)}") from e
    finally:
        del iq_signal, psd, psd_db
        gc.collect()

def process_difference(
    output_dir: Path,
    psd_cache: list,
    fft_size: int,
    center_freq: float,
    dpi: int
):
    """生成差分频谱图"""
    with tqdm(total=len(psd_cache)-1, desc="生成差分图", unit="pair") as diff_pbar:
        for i in range(len(psd_cache)-1):
            try:
                # 加载相邻文件数据
                prev_name, prev_data = psd_cache[i]
                next_name, next_data = psd_cache[i+1]
                
                # 确保频率轴一致
                if not np.allclose(prev_data[0], next_data[0]):
                    raise ValueError("频率轴不匹配，无法计算差分")

                # 计算差分
                diff_psd = next_data[1] - prev_data[1]

                # 绘图配置
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
                
                # 原始频谱对比
                ax1.plot(prev_data[0]/1e6, prev_data[1], label=prev_name, alpha=0.7)
                ax1.plot(next_data[0]/1e6, next_data[1], label=next_name, alpha=0.7)
                ax1.set_title("Original Spec", fontsize=10)
                ax1.legend()
                
                # 差分频谱
                ax2.plot(prev_data[0]/1e6, diff_psd, color='red', linewidth=0.8)
                ax2.set_title("Diff Spec", fontsize=10)
                
                # 公共设置
                for ax in [ax1, ax2]:
                    ax.set_xlim((center_freq - 1e6)/1e6, (center_freq + 1e6)/1e6)  # 显示中心频率±1MHz
                    ax.set_xlabel("Frequency [MHz]", fontsize=9)
                    ax.set_ylabel("ΔPSD [dB]", fontsize=9)
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                output_file = output_dir / f"diff_{prev_name}_vs_{next_name}.png"
                plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                
            except Exception as e:
                print(f"\n差分处理失败 {prev_name} vs {next_name}: {str(e)}")
            finally:
                diff_pbar.update(1)
                gc.collect()

if __name__ == "__main__":
    batch_iq_analysis(
        input_dir="A:/Radioation/20230508/IQ/2023_05_09", #数据文件夹路径
        output_dir="F:/2", #处理结果输出路径
        fft_size=32768,
        overlap=16384,
        window_type='blackman',
        center_freq=1419525000,
        dpi=200,
        max_memory_usage=85,
        enable_diff=True
    )
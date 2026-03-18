#ПРЕОБРАЗОВАНИЕ SNP данныХ (представленныХ в формате HW-кодирования) в CWT-ИЗОБРАЖЕНИЯ

import os
import pandas as pd
from PIL import Image
import numpy as np
import pywt
import matplotlib.pyplot as plt
from tqdm import tqdm


class SNPtoCWT:
    def __init__(self, csv_path, output_dir='cwt_images', plots_dir='cwt_plots'):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.plots_dir = plots_dir
        self.df_snps = None
        self.image_paths = []
        self.plot_paths = []

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

    def load_hw_data(self):
        print("Загрузка HW-кодированных SNP данных...")
        self.df_snps = pd.read_csv(self.csv_path, index_col=0)
        print(f"Загружено: {self.df_snps.shape[0]} образцов, {self.df_snps.shape[1]} SNP")
        return self.df_snps

    def preprocess_signal(self, snp_signal):
        snp_clean = np.nan_to_num(snp_signal, nan=0.0)
        return snp_clean

    def cwt_transform(self, snp_signal, wavelet='cmor', scales=None):  # morl
        signal_length = len(snp_signal)

        max_scales = 256

        if scales is None:
            min_scale = 1
            max_scale = min(signal_length // 4, max_scales)
            num_scales = min(64, max_scale - min_scale)
            scales = np.arange(min_scale, max_scale + 1)
            #print("scales = ", scales)

        try:
            coefficients, frequencies = pywt.cwt(snp_signal, scales, wavelet)

            # Извлекаем компоненты
            amplitude = np.abs(coefficients)  # Амплитуда
            phase = np.angle(coefficients)  # Фаза (-π до π)
            real_part = np.real(coefficients)  # Действительная часть
            imag_part = np.imag(coefficients)  # Мнимая часть

            return {
                'coefficients': coefficients,
                'frequencies': frequencies,
                'amplitude': amplitude,
                'phase': phase,
                'real': real_part,
                'imaginary': imag_part
            }
        except Exception as e:
            print(f"Ошибка CWT: {e}")
            zero_array = np.zeros((len(scales), signal_length))
            return {
                'coefficients': zero_array,
                'frequencies': np.array(frequencies),
                'amplitude': zero_array,
                'phase': zero_array,
                'real': zero_array,
                'imaginary': zero_array
            }

    '''def create_cwt_image(self, coefficients, sample_id,
                        cmap='viridis', dpi=300):

        height, width = coefficients.shape

        fig_width = width / dpi  # Ширина в дюймах = пиксели / DPI
        fig_height = height / dpi

        fig = plt.figure(figsize=(fig_width, fig_height), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        im = ax.imshow(np.abs(coefficients),
                       aspect='auto',
                       cmap=cmap,
                       interpolation='nearest') #без интерполяции

        output_path = os.path.join(self.output_dir, f'{sample_id}_cwt.png')
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                    pad_inches=0, facecolor='black')
        plt.close(fig)

        return output_path'''

    def create_cwt_image(self, coefficients, sample_id):
        """
        Создание grayscale изображения CWT с помощью PIL
        """
        # Берем амплитуду коэффициентов
        coeff_abs = np.abs(coefficients)

        # Нормализуем в диапазон [0, 1]
        coeff_normalized = (coeff_abs - np.min(coeff_abs)) / (np.max(coeff_abs) - np.min(coeff_abs))

        # Конвертируем в диапазон [0, 255] для grayscale
        coeff_uint8 = (coeff_normalized * 255).astype(np.uint8)

        # Создаем изображение
        height, width = coeff_uint8.shape
        image = Image.fromarray(coeff_uint8, mode='L')  # 'L' для grayscale

        # Сохраняем
        output_path = os.path.join(self.output_dir, f'{sample_id}_cwt.png')
        image.save(output_path, format='PNG', dpi=(300, 300))

        return output_path

    def plot_frequency_analysis(self, cwt_result, sample_id, snp_signal):
        """
        Построение графиков частотного анализа для CWT коэффициентов
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Частотный анализ CWT - Образец {sample_id}', fontsize=16, fontweight='bold')

        # 1. Исходный SNP сигнал
        axes[0, 0].plot(snp_signal, 'b-', linewidth=1)
        axes[0, 0].set_title('Исходный SNP сигнал')
        axes[0, 0].set_xlabel('Позиция SNP')
        axes[0, 0].set_ylabel('Значение')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Амплитудный спектр (среднее по времени) - С ТОЧКАМИ
        mean_amplitude = np.mean(cwt_result['amplitude'], axis=1)
        axes[0, 1].semilogy(cwt_result['frequencies'], mean_amplitude, 'r-', linewidth=2)
        axes[0, 1].semilogy(cwt_result['frequencies'], mean_amplitude, 'ro', markersize=4, alpha=0.6)
        axes[0, 1].set_title('Средняя амплитуда по частотам')
        axes[0, 1].set_xlabel('Частота')
        axes[0, 1].set_ylabel('Амплитуда')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xscale('log')

        # 3. Максимальная амплитуда по частотам - С ТОЧКАМИ
        max_amplitude = np.max(cwt_result['amplitude'], axis=1)
        axes[0, 2].semilogy(cwt_result['frequencies'], max_amplitude, 'g-', linewidth=2)
        axes[0, 2].semilogy(cwt_result['frequencies'], max_amplitude, 'go', markersize=4, alpha=0.6)
        axes[0, 2].set_title('Максимальная амплитуда по частотам')
        axes[0, 2].set_xlabel('Частота')
        axes[0, 2].set_ylabel('Амплитуда')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_xscale('log')

        # 4. Фазовый профиль (среднее по времени)
        mean_phase = np.mean(cwt_result['phase'], axis=1)
        axes[1, 0].plot(cwt_result['frequencies'], mean_phase, 'm-', linewidth=2)
        axes[1, 0].set_title('Средняя фаза по частотам')
        axes[1, 0].set_xlabel('Частота')
        axes[1, 0].set_ylabel('Фаза (рад)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xscale('log')

        # 5. Действительная и мнимая части (средние) - С ТОЧКАМИ
        mean_real = np.mean(cwt_result['real'], axis=1)
        mean_imag = np.mean(cwt_result['imaginary'], axis=1)
        axes[1, 1].plot(cwt_result['frequencies'], mean_real, 'c-', linewidth=2, label='Re')
        axes[1, 1].plot(cwt_result['frequencies'], mean_real, 'co', markersize=4, alpha=0.6)
        axes[1, 1].plot(cwt_result['frequencies'], mean_imag, 'y-', linewidth=2, label='Im')
        axes[1, 1].plot(cwt_result['frequencies'], mean_imag, 'yo', markersize=4, alpha=0.6)
        axes[1, 1].set_title('Действительная и мнимая части')
        axes[1, 1].set_xlabel('Частота')
        axes[1, 1].set_ylabel('Значение')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xscale('log')

        # 6. Энергия по частотам - С ТОЧКАМИ
        energy = np.sum(cwt_result['amplitude'] ** 2, axis=1)
        axes[1, 2].semilogy(cwt_result['frequencies'], energy, 'k-', linewidth=2)
        axes[1, 2].semilogy(cwt_result['frequencies'], energy, 'ko', markersize=4, alpha=0.6)
        axes[1, 2].set_title('Энергия по частотам')
        axes[1, 2].set_xlabel('Частота')
        axes[1, 2].set_ylabel('Энергия')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_xscale('log')

        plt.tight_layout()

        # Сохраняем график
        plot_path = os.path.join(self.plots_dir, f'{sample_id}_frequency_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return plot_path

    def plot_time_frequency_analysis(self, cwt_result, sample_id):
        """
        Дополнительный анализ временно-частотных характеристик
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Временно-частотный анализ - Образец {sample_id}', fontsize=14)

        # 1. Спектрограмма амплитуды
        im1 = axes[0, 0].imshow(cwt_result['amplitude'], aspect='auto', cmap='viridis',
                                extent=[0, cwt_result['amplitude'].shape[1],
                                        cwt_result['frequencies'][-1], cwt_result['frequencies'][0]])
        axes[0, 0].set_title('Спектрограмма амплитуды')
        axes[0, 0].set_ylabel('Частота')
        axes[0, 0].set_yscale('log')
        plt.colorbar(im1, ax=axes[0, 0])

        # 2. Спектрограмма фазы
        im2 = axes[0, 1].imshow(cwt_result['phase'], aspect='auto', cmap='hsv',
                                extent=[0, cwt_result['phase'].shape[1],
                                        cwt_result['frequencies'][-1], cwt_result['frequencies'][0]],
                                vmin=-np.pi, vmax=np.pi)
        axes[0, 1].set_title('Спектрограмма фазы')
        axes[0, 1].set_yscale('log')
        plt.colorbar(im2, ax=axes[0, 1])

        # 3. Профиль доминирующих частот
        dominant_freq_idx = np.argmax(cwt_result['amplitude'], axis=0)
        dominant_frequencies = cwt_result['frequencies'][dominant_freq_idx]
        axes[1, 0].plot(dominant_frequencies, 'b-', alpha=0.7)
        axes[1, 0].set_title('Доминирующие частоты во времени')
        axes[1, 0].set_xlabel('Время')
        axes[1, 0].set_ylabel('Частота')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Распределение энергии по масштабам
        scale_energy = np.sum(cwt_result['amplitude'] ** 2, axis=1)
        axes[1, 1].plot(cwt_result['frequencies'], scale_energy, 'r-', linewidth=2)
        axes[1, 1].set_title('Распределение энергии по частотам')
        axes[1, 1].set_xlabel('Частота')
        axes[1, 1].set_ylabel('Энергия')
        axes[1, 1].set_xscale('log')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = os.path.join(self.plots_dir, f'{sample_id}_time_frequency.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return plot_path

    def process_single_sample(self, sample_id, snp_signal, wavelet='morl', create_plots=True):
        try:
            processed_signal = self.preprocess_signal(snp_signal)
            cwt_result = self.cwt_transform(processed_signal, wavelet=wavelet)

            # Создаем CWT изображение
            image_path = self.create_cwt_image(cwt_result['coefficients'], sample_id)

            plot_paths = []
            if create_plots:
                # Создаем графики анализа
                freq_plot_path = self.plot_frequency_analysis(cwt_result, sample_id, processed_signal)
                time_freq_plot_path = self.plot_time_frequency_analysis(cwt_result, sample_id)
                plot_paths = [freq_plot_path, time_freq_plot_path]

            return image_path, plot_paths, cwt_result

        except Exception as e:
            print(f"Ошибка обработки образца {sample_id}: {e}")
            return None, [], None

    def batch_convert(self, wavelet='morl', create_plots=True,
                      max_samples=None, progress_bar=True):

        if self.df_snps is None:
            self.load_hw_data()

        if max_samples:
            df_to_process = self.df_snps.head(max_samples)
        else:
            df_to_process = self.df_snps

        print(f"Начинаю преобразование {len(df_to_process)} образцов...")

        samples_to_process = tqdm(df_to_process.iterrows(), total=len(df_to_process)) \
            if progress_bar else df_to_process.iterrows()

        results = []
        plot_results = []

        for sample_id, snp_signal in samples_to_process:
            image_path, plot_paths, cwt_result = self.process_single_sample(
                sample_id, snp_signal.values, wavelet, create_plots
            )

            if image_path:
                result_entry = {
                    'sample_id': sample_id,
                    'image_path': image_path,
                    'plots_dir': self.plots_dir
                }

                if create_plots and plot_paths:
                    result_entry['frequency_plot'] = plot_paths[0]
                    result_entry['time_frequency_plot'] = plot_paths[1]

                results.append(result_entry)
                plot_results.append({
                    'sample_id': sample_id,
                    'cwt_result': cwt_result,
                    'signal': snp_signal.values
                })

        self.image_paths = results
        print(f"Успешно обработано: {len(results)} образцов")

        return results, plot_results



def main():
    converter = SNPtoCWT(
        csv_path='hw_encoded_snps.csv',
        output_dir='snp_cwt_images',
        plots_dir='snp_cwt_plots'
    )

    results, plot_results = converter.batch_convert(
        wavelet='morl',
        create_plots=True,
        max_samples=None,
        progress_bar=True
    )

    df_results = pd.DataFrame(results)
    df_results.to_csv('cwt_image_paths.csv', index=False)
    print("Список путей к изображениям сохранен: cwt_image_paths.csv")

    if results:
        first_result = results[0]
        print(f"Первый образец: {first_result['sample_id']}")
        print(f"Изображение CWT: {first_result['image_path']}")

        if 'frequency_plot' in first_result:
            print(f"График частотного анализа: {first_result['frequency_plot']}")
            print(f"График временно-частотного анализа: {first_result['time_frequency_plot']}")

        # Показываем пример графика
        if plot_results:
            first_plot_data = plot_results[0]
            converter.plot_frequency_analysis(
                first_plot_data['cwt_result'],
                first_plot_data['sample_id'],
                first_plot_data['signal']
            )

            # Загружаем и показываем CWT изображение
            img = plt.imread(first_result['image_path'])
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.imshow(img, cmap='gray')
            plt.title(f"CWT изображение\n{first_result['sample_id']}")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.plot(first_plot_data['signal'])
            plt.title('Исходный SNP сигнал')
            plt.xlabel('Позиция SNP')
            plt.ylabel('Значение')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()

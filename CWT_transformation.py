import os
import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
from tqdm import tqdm


class SNPtoCWT:
    def __init__(self, csv_path, output_dir='cwt_images'):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.df_snps = None
        self.image_paths = []

        os.makedirs(output_dir, exist_ok=True)

    def load_hw_data(self):
        print("Загрузка HW-кодированных SNP данных...")
        self.df_snps = pd.read_csv(self.csv_path, index_col=0)
        print(f"Загружено: {self.df_snps.shape[0]} образцов, {self.df_snps.shape[1]} SNP")
        return self.df_snps

    def preprocess_signal(self, snp_signal):
        snp_clean = np.nan_to_num(snp_signal, nan=0.0)
        return snp_clean

    def cwt_transform(self, snp_signal, wavelet='morl', scales=None):
        signal_length = len(snp_signal)

        if scales is None:
            min_scale = 1
            max_scale = min(signal_length // 4, 128)
            num_scales = min(64, max_scale - min_scale)
            scales = np.arange(min_scale, max_scale + 1,
                               max(1, (max_scale - min_scale) // num_scales))

        try:
            coefficients, frequencies = pywt.cwt(snp_signal, scales, wavelet)
            return coefficients, frequencies
        except Exception as e:
            print(f"Ошибка CWT: {e}")
            return np.zeros((len(scales), signal_length)), scales

    def create_cwt_image(self, coefficients, sample_id,
                        cmap='viridis', figsize=(8, 6), dpi=100):
        fig = plt.figure(figsize=figsize, frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        im = ax.imshow(np.abs(coefficients),
                       aspect='auto',
                       cmap=cmap,
                       interpolation='bilinear')

        output_path = os.path.join(self.output_dir, f'{sample_id}_cwt.png')
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                    pad_inches=0, facecolor='black')
        plt.close(fig)

        return output_path

    def process_single_sample(self, sample_id, snp_signal,
                              create_grayscale=True, wavelet='morl'):
        try:
            processed_signal = self.preprocess_signal(snp_signal)

            coefficients, frequencies = self.cwt_transform(processed_signal, wavelet=wavelet)

            color_path = self.create_cwt_image(coefficients, sample_id, cmap='viridis')

            return color_path

        except Exception as e:
            print(f"Ошибка обработки образца {sample_id}: {e}")
            return None, None

    def batch_convert(self, wavelet='morl', create_grayscale=True,
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

        for sample_id, snp_signal in samples_to_process:
            color_path = self.process_single_sample(
                sample_id, snp_signal.values, create_grayscale, wavelet
            )

            if color_path:
                results.append({
                    'sample_id': sample_id,
                    'color_image': color_path
                })

        self.image_paths = results
        print(f"Успешно обработано: {len(results)} образцов")

        return results


def main():
    converter = SNPtoCWT(
        csv_path='hw_encoded_snps.csv',
        output_dir='snp_cwt_images'
    )

    results = converter.batch_convert(
        wavelet='morl',
        create_grayscale=False,
        max_samples=None,
        progress_bar=True
    )

    df_results = pd.DataFrame(results)
    df_results.to_csv('cwt_image_paths.csv', index=False)
    print("Список путей к изображениям сохранен: cwt_image_paths.csv")

    if results:
        first_image = results[0]['color_image']
        print(f"Первое изображение: {first_image}")

        img = plt.imread(first_image)
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.title(f"CWT изображение образца {results[0]['sample_id']}")
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    main()


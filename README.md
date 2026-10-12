# Tugas 05: Transfer Learning untuk Klasifikasi Citra dengan Data Terbatas

Peter Hansen (2802515910)

Perbandingan *from scratch*, *feature extraction* dan *fine-tuning* pada ResNet-18, uji kontrol *data augmentation*, dan perbandingan dengan EfficientNet-B0 pada Oxford-IIIT Pet (10 kelas: 5 kucing, 5 anjing; 1996 citra).

## Isi repositori

```
Tugas05_Transfer_Learning.ipynb   notebook utama (sudah berisi output)
requirements.txt
README.md
results/                          log mentah dan gambar hasil (dari hasil_tugas05.zip)
  raw_epoch_log.csv               log per epoch semua percobaan
  raw_summary.csv                 ringkasan metrik tiap percobaan
  raw_metrics.json                konfigurasi (seed, batch size, dll.) + ringkasan + riwayat
  tabel_konsolidasi.md            tabel konsolidasi (C1)
  overfitting_gap.csv             selisih train dan validation
  failure_cases.csv               citra test yang salah diprediksi (C3)
  figures/                        kurva belajar, confusion matrix, failure analysis, Grad-CAM
```

## Cara menjalankan (Google Colab)

1. Buka notebook di Colab dan pilih `Runtime > Change runtime type > T4 GPU`.
2. `Runtime > Run all`. Dataset (sekitar 800 MB) diunduh otomatis oleh `torchvision.datasets.OxfordIIITPet`.
3. Sel terakhir membuat `hasil_tugas05.zip` berisi folder `results`.

Notebook Colab: https://colab.research.google.com/drive/1r4ElD60ZIBxeKCzp8mZyA9tVeCTgUlZb?usp=sharing

## Protokol

| Item | Nilai |
|---|---|
| Model utama | ResNet-18, bobot ImageNet1K_V1 |
| Input | 224x224, normalisasi ImageNet |
| Optimizer dan loss | AdamW (weight decay 0,01), CrossEntropyLoss |
| Batch size | 32 |
| Early stopping | validation loss, patience 4, maksimum 30 epoch, bobot terbaik dipulihkan |
| Seed | 42 (Python, NumPy, PyTorch) |
| Split | stratified 70/15/15 (1397 / 299 / 300 citra) |

| Kode | Kondisi | Learning rate |
|---|---|---|
| B1 | ResNet-18 dari nol | 1e-3 |
| B2 | Feature extraction (hanya `fc`, 5.130 parameter) | 1e-3 |
| B3 | Fine-tuning `layer4` + `fc` (8.398.858 parameter) | layer4 1e-4, fc 1e-3 |
| B4 | B3 tanpa augmentasi (B4a) dan dengan augmentasi (B4b = B3) | sama dengan B3 |
| B5 | EfficientNet-B0 dengan strategi terbaik antara B2 dan B3 (dipilih dari akurasi validation) | sama dengan strategi terpilih |

## Catatan

- Lapisan yang dibekukan dipaksa ke mode `eval()` supaya running statistics BatchNorm tidak berubah.
- Augmentasi hanya dipakai pada data train. Validation dan test hanya di-resize, di-crop dan dinormalisasi.
- Model terbaik untuk failure analysis dipilih dari akurasi validation, bukan test.
- Hasil di GPU bisa sedikit berbeda antar mesin atau versi library walau seed sama. Studi ini memakai satu seed.

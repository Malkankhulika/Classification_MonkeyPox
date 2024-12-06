# <h1 align="center">Classification_MonkeyPox</h1>
<p align="center">Khulika Malkan</p>
<p align="center">2311110057</p>

## Features

- [Gambaran Umum MonkeyPoX](#GambaranumumMonkeyPox)
- [About Dataset](#AboutDataset)
- [Code](#Code)
- [Output](#Output)
- [Kesimpulan](#Kesimpulan)

## Gambaran Umum MonkeyPox
![image](https://github.com/user-attachments/assets/caecf5dc-1c1a-42b5-87ca-7684067f2f56)

Pada Mei 2022, wabah monkeypox, sebuah penyakit virus, dikonfirmasi terjadi. Kasus pertama ditemukan di Inggris pada 6 Mei 2022, pada individu yang memiliki riwayat perjalanan ke Nigeria, tempat penyakit ini endemik. Wabah ini menandai pertama kalinya monkeypox menyebar secara luas di luar Afrika Tengah dan Barat. Mulai 18 Mei, kasus terus dilaporkan dari berbagai negara dan wilayah, terutama di Eropa, tetapi juga di Amerika Utara dan Selatan, Asia, Afrika, serta Oseania. Pada 23 Juli, Direktur Jenderal Organisasi Kesehatan Dunia (WHO), Tedros Adhanom Ghebreyesus, menyatakan wabah ini sebagai darurat kesehatan masyarakat yang menjadi perhatian internasional (PHEIC). Hingga 12 Agustus, tercatat 34.448 kasus terkonfirmasi di lebih dari 82 negara, dengan sebagian besar negara mengalami kasus pertama mereka.


Monkeypox adalah infeksi virus yang biasanya muncul sekitar satu hingga dua minggu setelah paparan. Gejalanya diawali dengan demam dan gejala tidak spesifik lainnya, diikuti dengan munculnya ruam dan lesi yang berlangsung selama 2–4 minggu sebelum kering, mengeras, dan terkelupas. Meski dapat menyebabkan banyak lesi, pada wabah saat ini beberapa pasien hanya mengalami satu lesi, biasanya di mulut atau area genital, sehingga lebih sulit dibedakan dari infeksi lainnya. Dalam infeksi sebelum wabah ini, sekitar 1–3 persen penderita yang terdiagnosis diketahui meninggal jika tidak mendapatkan pengobatan. Kasus pada anak-anak dan individu dengan gangguan kekebalan lebih berisiko menjadi parah.


## About Dataset
![image](https://github.com/user-attachments/assets/2b8fa5bf-42ee-41fa-b6b5-4f2f4235b177)

Dataset ini terdiri dari dua folder utama:
1.	Original Images:
Folder ini berisi subfolder bernama "FOLDS," yang terdiri dari lima fold (fold1 hingga fold5) yang digunakan untuk validasi silang 5-fold. Setiap fold memiliki subfolder terpisah untuk dataset test, train, dan validation.

3.	Augmented Images:
Folder ini memiliki subfolder bernama "FOLDS_AUG," yang berisi gambar augmented dari set train di setiap fold yang ada dalam subfolder "FOLDS" di "Original Images." Proses augmentasi ini meningkatkan jumlah gambar hingga sekitar 14 kali lipat dari jumlah aslinya.


## Full code Screenshot:
1.  PROGRAM/Utils/getData.py
   
![image](https://github.com/user-attachments/assets/bcf800eb-5123-437e-8bc9-9b8ac9fe6748)



2. PROGRAM/model/Train.py

![image](https://github.com/user-attachments/assets/ea5e2724-a740-4736-b61c-d7acdb5accba)



3. PROGRAM/model/Test.py
   
![image](https://github.com/user-attachments/assets/a04993c7-8ac2-4a84-8f55-f39e2636050b)


## Output & Interpretasi
1. getData
![image](https://github.com/user-attachments/assets/64cd57eb-3fb4-4a29-b2de-a8ae3393b831)

Output di atas memberikan gambaran lengkap tentang distribusi data dalam dataset yang dimuat.
   
2. Train
![image](https://github.com/user-attachments/assets/7927d1be-4a9d-4cf4-8f84-e294367f4128)

Log Epoch
Epoch 1: Model mulai belajar, tetapi masih kesulitan memahami data. Validation accuracy mendekati train accuracy, menunjukkan generalisasi awal yang baik.
Epoch 5: Model berhasil mencapai train accuracy yang tinggi (83%) dengan validation accuracy 78%. Performa validasi yang mendekati pelatihan menunjukkan model tidak terlalu overfitting.

![image](https://github.com/user-attachments/assets/0fa8dc38-0045-47a3-b674-706161641f63)

Grafik Loss
Training Loss (Pink): Menurun secara konsisten, menunjukkan bahwa model semakin baik dalam meminimalkan kesalahan pada data pelatihan.
Validation Loss (Ungu): Menurun dengan pola serupa, tetapi memiliki fluktuasi kecil, yang merupakan pola umum validasi.

3. Test
![image](https://github.com/user-attachments/assets/26b0a820-f049-400e-9eba-21f31104b4e9)

![image](https://github.com/user-attachments/assets/e8d8240e-05d4-4542-910c-360005a97a43)


## Referensi
[1] https://builtin.com/machine-learning/mobilenet

[2] https://en.wikipedia.org/wiki/2022%E2%80%932023_mpox_outbreak

[3] https://keras.io/api/applications/mobilenet/

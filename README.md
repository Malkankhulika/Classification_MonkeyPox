# <h1 align="center">Laporan Praktikum Modul Array</h1>
<p align="center">Khulika Malkan</p>

## Features

- [Dasar Teori](#dasar-teori)
- [Guided](#guided)
- [Unguided](#unguided)
- [Kesimpulan](#kesimpulan)
- [Referensi](#referensi)

## Dasar Teori

Array (larik) ialah penampung sejumlah data sejenis (homogen) yang menggunakan satu identifier (pengenal). Masing-masing elemen larik diakses menggunakan indeks (subscript) dari nol sampai n-1 (n menyatakan jumlah elemen larik). Pengolahan data larik harus per elemen. Elemen larik dapat diakses langsung (acak). Tujuannya adalah untuk memanipulasi elemen keempat tidak harus melalui elemen pertama, kedua, dan ketiga. Berdasarkan banyaknya indeks, larik dibagi menjadi larik satu dimensi dan multidimensi (dua dimensi dan tiga dimensi)[1].

### 1. Array Satu Dimensi
Array satu dimensi adalah kumpulan elemen yang bersebelahan di mana individu elemen diidentifikasi oleh subskrip integer unik yang dimulai dengan nol. Sekali array dibuat, ukurannya tidak dapat diubah[2]. Setiap elemen di dalam array memiliki sebuah indeks atau nomor yang digunakan untuk mengakses elemen tersebut. Indeks dimulai dari 0 dan berakhir pada jumlah elemen dikurangi satu. 

Contohnya, sebuah array satu dimensi yang berisi bilangan bulat {1, 2, 3, 4, 5} memiliki lima elemen dan indeksnya dimulai dari 0. Indeks 0 merujuk pada elemen pertama, indeks 1 merujuk pada elemen kedua, dan seterusnya hingga indeks 4 merujuk pada elemen kelima

### 2. Array Dua Dimensi
Array dua demensi yakni suatu kumpulan elemen yang dapat dilakukan penyimpanan secara terurut di suatu tempat yang terbagi menjadi dua 2 unsur yakni baris dan kolom[2]. Dalam array dua dimensi, setiap elemen memiliki dua indeks, yaitu indeks baris dan indeks kolom. Indeks baris menunjukkan posisi elemen dalam baris, sementara indeks kolom menunjukkan posisi elemen dalam kolom

### 3. Array Multidimensi
Array multidimensi memiliki kesamaan dengan array satu dimensi dan dua dimensi, namun memiliki kapasitas memori yang lebih besar. Array ini digunakan untuk merepresentasikan array dengan dimensi lebih dari dua atau array yang memiliki lebih dari dua indeks, seperti array tiga dimensi, array empat dimensi, array lima dimensi, dan seterusnya.

## Guided 

### 1. Program Input Array Tiga Dimensi

```C++
#include <iostream>
using namespace std;

//PROGRAM INPUT ARRAY 3 DIMENSI
int main() {
	//Deklarasi Array
	int arr[2][3][3];

	//Input elemen
	for (int x = 0; x < 2; x++)
	{
		for (int y = 0; y < 3; y++)
		{
			for (int z = 0; z < 3; z++)
			{
				cout << "Input Array[" << x << "][" << y << "][" << z << "] = ";
				cin >> arr[x][y][z];
			}
		}
		cout << endl;
	}

	//Output array
	for (int x = 0; x < 2; x++)
	{
		for (int y = 0; y < 3; y++)
		{
			for (int z = 0; z < 3; z++)
			{
				cout << "Data Array[" << x << "][" << y << "][" << z << "] = " << arr[x][y][z] << endl;
			}
		} 
	}

	cout << endl;
	//Tampilan array
	for (int x = 0; x < 2; x++)
	{
		for (int y = 0; y < 3; y++)
		{
			for (int z = 0; z < 3; z++)
			{
				cout << arr[x][y][z] << ends;
			}
			cout << endl;
		}
		cout << endl;
	}
}
```
Kode di atas digunakan untuk mencetak array tiga dimensi. Pertama membuat variabel `arr` dengan mengisi x, y, dan z dengan `[2][3][4]`. Lalu kita membuat nested looping untuk menginputkan angka kedalam elemen. Lalu setelah itu, kita membuat nested looping kembali untuk menampilkan angka pada setiap elemen yang sudah diinputkan tadi. Setelah itu, kita membuat kembali nested loop untuk menampilkan angka pada elemen, namun kali ini hanya ditampilkan nilainya saja tanpa perlu menampilkan setiap angka dan elemennya.

### 2. Program Mencari Nilai Maksimal pada Array

```C++
#include <iostream>
using namespace std;

int main() {
	int maks, a, lokasi;
	cout << "Masukkan panjang array: ";
	cin >> a;

	if (a <= 0) {
		cout << "Panjang array harus lebih besar dari 0." << endl;
		return 1; //Return 1 to indicate error
	}

	int array[a];
	cout << "Masukkan " << a << " angka\n";

	for (int i = 0; i < a; i++){
		cout << "Array ke-" << (i+ 1) << ": ";
		cin >> array[i];
	}

	maks = array[0];
	lokasi = 0;

	for (int i = 1; i < a; i++){
		if (array[i] > maks) {
			maks = array[i];
			lokasi = i;
		}
	}

	cout << "Nilai maksimum adalah " << maks << " berada di Array ke " << (lokasi + 1) << endl;

	return 0; //Return 0 to indicate successful execution 
}
```
Kode di atas digunakan untuk mencari nilai tertinggi atau nilai maksimum pada suatu array. Pertama kita mendeklarasikan dahulu variabel beserta tipe datanya. Lalu setelah itu kita memasukkan panjang array. Dibawahnya terdapat `if` yang digunakan untuk memastikan user tidak menginputkan panjang array dengan nilai 0. Setelah itu kita memasukan nilai kedalam setiap elemen sesuai dengan panjang array yang sudah diinputkan tadi. Kemudian kita membuat looping yang digunakan untuk menampilkan setiap nilai yang sudah diinputkan ke dalam elemen dalam array. 

Kemudian kita akan mencari nilai maksimumnya dengan cara kita membuat variabel `maks` dan `lokasi`. Variabel `maks` berisi nilai array pada elemen pertama, dan variabel `lokasi` berisi nilai 0. Lalu kita membuat looping yang di dalamnya mengandung `if` yang digunakan untuk membandingkan array pada elemen tertentu dengan variabel `maks`. Jika array pada suatu elemen lebih besar daripada variabel `maks` yang berisikan nilai array pada elemen pertama, maka otomatis variabel `maks` akan diisi nilainya sesuai dengan nilai arraynya. Lalu variabel `lokasi` yang nilai awalnya di set pada angka 0 maka otomatis akan diganti dengan elemen pada array tersebut. Hal itu dilakukan secara berulang hingga menemukan angka yang paling besar atau maksimum pada suatu array. Setelah ditemukan, maka kita bisa tampilkan menggunakan fungsi `cout` nilai maksimumnya dan indeks ke berapa nilai tersebut.

## Unguided 

### 1. Buatlah program untuk menampilkan Output seperti berikut dengan data yang diinputkan oleh user!

```C++
#include <iostream> //input preprocessor directive
using namespace std; //supaya tidak perlu menggunakan std:: disetiap input object or function

int main() //main program
{
    int panjang; //mendeklarasikan variabel panjang dengan tipe data integer
    cout << "Masukkan panjang array: "; //input panjang array yang akan dibuat
    cin >> panjang;
    int arr[panjang]; //mendeklarasikan array sesuai dengan input panjang array dari user

    cout << "Masukkan " << panjang << " angka\n"; //menginputkan nilai ke dalam array yang sudah di inputkan
    for (int i = 0; i < panjang; i++)
    {
        cout << "Array ke - " << i << ": "; //input elemen array 
        cin >> arr[i];
    }
    cout << endl;

    cout << "Data Array : "; //output elemen array
    for (int i = 0; i < panjang; i++)
    {
        cout << arr[i] << " ";
    }
    cout << endl;

    cout << "Nomor Genap : "; //menampilkan angka genap
    for (int i = 0; i < panjang; i++)
    {
        if (arr[i] % 2 == 0)
        {
            cout << arr[i] << ", ";
        }
    }
    cout << endl;

    cout << "Nomor Ganjil : "; //menampilkan angka genap
    for (int i = 0; i < panjang; i++)
    {
        if (arr[i] % 2 != 0)
        {
            cout << arr[i] << ", ";
        }
    }
}
```
#### Output:
![unguided1](https://github.com/agungibr/Struktur-Data-Assignment/assets/91455543/204e19e5-6645-41a4-9c26-86508ff2d693)

Kode di atas digunakan untuk menampilkan angka ganjil dan angka genap pada suatu array. Pertama kita mendeklarasikan variabel `panjang` dengan tipe data integer. Lalu kita membuat fungsi untuk menginputkan panjang array. Setelah itu, kita mengisikan nilai ke dalam array yang sudah dibuat. Setelah diisikan kita menampilkan seluruh nilai beserta dengan indeksnya menggunakan looping `for`. Kemudian kita akan memfilter mana yang angka ganjil dan mana yang angka genap. Untuk angka genap kita menggunakan cara dengan nilai pada array modulus 2, jika hasil modulus sama dengan 0 maka angka tersebut masuk kedalam angka genap, namun jika hasil modulus tidak sama dengan 0, maka angka tersebut masuk kedalam angka ganjil.

#### Full code Screenshot:
![fullcode1](https://github.com/agungibr/Struktur-Data-Assignment/assets/91455543/89f41a4f-81a1-40e7-866b-f0820dcd1eea)


### 2. Buatlah program Input array tiga dimensi (seperti pada guided) tetapi jumlah atau ukuran elemennya diinputkan oleh user!

```C++
#include <iostream> //input preprocessor directive
using namespace std; //supaya tidak perlu menggunakan std:: disetiap input object or function

int main() //main program
{
    int x, y, z; //mendeklarasikan variabel x, y, z dengan tipe data integer
    
    cout << "Input jumlah matrix: "; //input panjang matrix
    cin >> x;
    cout << "Input jumlah baris: "; //input panjang baris
    cin >> y;
    cout << "Input jumlah kolom: "; //input panjang kolom
    cin >> z;
    cout << endl;
    
    int arr[x][y][z]; //mendeklarasikan array sesuai input

    for (int i = 0; i < x; i++) //looping input elemen
    {
        for (int j = 0; j < y; j++)
        {
            for (int k = 0; k < z; k++)
            {
                cout << "Input Array[" << i << "][" << j << "][" << k << "] = "; //input nilai setiap elemen
                cin >> arr[i][j][k];
            }
        }
        cout << endl;
    }

    for (int i = 0; i < x; i++) //looping untuk menampilkan inputan array
    {
        for (int j = 0; j < y; j++)
        {
            for (int k = 0; k < z; k++)
            {
                cout << "Data Array[" << i << "][" << j << "][" << k << "] = " << arr[i][j][k] << endl; //menampilkan inputan array
            }
        }
    }
    cout << endl;

    for (int i = 0; i < x; i++) //looping untuk menampilkan hasil array
    {
        for (int j = 0; j < y; j++)
        {
            for (int k = 0; k < z; k++)
            {
                cout << arr[i][j][k] << " "; //menampilkan hasil array
            }
            cout << endl;
        }
        cout << endl;
    }
}
```
#### Output:
![unguided2](https://github.com/agungibr/Struktur-Data-Assignment/assets/91455543/3002da6b-207c-479d-9932-056123bc3067)

Kode di atas digunakan untuk mencetak array tiga dimensi dengan menggunakan fungsi input dari user. Pertama kita membuat dahulu fungsi inputannya. User bisa memasukkan jumlah matrix, jumlah baris, dan jumlah kolom. Lalu kita membuat nested looping untuk menginputkan angka kedalam elemen. Lalu setelah itu, kita membuat nested looping kembali untuk menampilkan angka pada setiap elemen yang sudah diinputkan tadi. Setelah itu, kita membuat kembali nested loop untuk menampilkan angka pada elemen, namun kali ini hanya ditampilkan nilainya saja tanpa perlu menampilkan setiap angka dan elemennya.

#### Full code Screenshot:
![fullcode2](https://github.com/agungibr/Struktur-Data-Assignment/assets/91455543/9b78084a-4f8f-482a-ba9d-712ede89b2a9)


### 3. Buatlah program menu untuk mencari nilai Maksimum, Minimum dan Nilai rata – rata dari suatu array dengan input yang dimasukan oleh user!

```C++
#include <iostream> //input preprocessor directive
using namespace std; //supaya tidak perlu menggunakan std:: disetiap input object or function

int main() { //main program
	int length, op, a, i; //mendeklarasikan variabel
	float average = 0, minimum = 99999, maximum = 0;

	cout << "Input jumlah array: "; //menginputkan panjang array
	cin >> length;
	int array[length];

	cout << "Input " << length << " angka kedalam array" << endl; //menginputkan angka kedalam elemen
	for (i = 0; i < length; ++i) { //looping menginputkan angka ke elemen sesuai jumlah array
        cout << "Array[" << i << "]: ";
        cin >> array[i];
    }

    for (i = 0; i < length; ++i) { //mencari nilai maximum
    	if (array[i] > maximum) { //jika nilai array lebih kecil dari variabel maximum
    		maximum = array[i]; //maka nilai maximum diubah menjadi nilai array
    	}
    }

    for (i = 0; i < length; ++i) { //mencari nilai minimum
    	if (array[i] < minimum) { //jika nilai array lebih besar dari variabel minimum
    		minimum = array[i]; //maka nilai minimum diubah menjadi nilai array
    	}
    }

    for (i = 0; i < length; ++i) { //mencari rata-rata
    	average += array[i]; //menjumlahkan seluruh elemen
    }
    float summary = average / length; //mencari rata-rata

    do //looping menu
    {
    	cout << "\n===== MENU =====" << endl;
        cout << "1. Nilai Maksimum" << endl;
        cout << "2. Nilai Minimum" << endl;
        cout << "3. Nilai Rata-rata" << endl;
        cout << "0. Exit" << endl;
        cout << "Enter your choice: ";
        cin >> op;

        switch(op)
        {
        case 1: //jika memilih 1 maka akan mencari nilai maksimum
        	cout << "\nNilai Maksimum: " << maximum << endl;
        	break;
        case 2: //jika memilih 2 maka akan mencari nilai minimum
        	cout << "\nNilai Minimum: " << minimum << endl;
            break;
        case 3: //jika memilih angka 3 maka akan mencari nilai rata-rata
        	cout << "\nNilai Rata-rata: " << summary << endl;
            break;
        case 0: //jika ingin keluar dari looping maka bisa menginputkan angka 0
        	cout << "\nThank you for using this program. Goodbye!" << endl;
            break;
        default: //error message jika menginputkan selain angka 1, 2, 3, dan 0
    		cout << "\nInvalid input. Please enter either '1', '2', '3', or '0'." << endl;
    		break;
        }
    } while (op != 0); //jika user menginputkan angka selain 0 maka program otomatis akan mengulang hingga user menginputkan angka 0 untuk mengakhiri loop

    return 0;
}
```
#### Output:
![unguided3](https://github.com/agungibr/Struktur-Data-Assignment/assets/91455543/2473e55a-07ea-4778-acd1-c60e1e31939a)

Kode di atas digunakan untuk mencetak program menu untuk mencari nilai Maksimum, Minimum dan Nilai rata – rata dari suatu array dengan input yang dimasukan oleh user. Pertama kita mendeklarasikan variabel yang akan digunakan. Disini untuk variabel `minimum` di set ke dalam `99999` karena variabel ini akan digunakan untuk pembanding dari nilai array. Kemudian kita membuat fungsi untuk menginputkan panjang array. Setelah menginputkan panjang array, kita perlu memasukan nilai ke dalam array sesuai dengan panjang array yang sudah diinputkan tadi. Lalu kita membuat fungsi untuk mencari nilai minimum, maksimum, dan rata-rata. Untuk minimum kita mencarinya dengan cara membandingkan nilai array dengan variabel `minimum`. Jika nilai pada array lebih kecil daripada variabel `minimum` maka variabel `minimum` akan diubah sesuai dengan nilai array. Kemudian hal tersebut diulang sampai menemukan nilai minimumnya. Hal tersebut juga sama jika kita ingin mencari nilai maksimumnya. 

Untuk mencari rata-rata kita menjumlahkan seluruh elemen dan kemudian dibagi dengan panjang array. Setelah membuat fungsi, kita perlu membuat menu untuk user. Dengan menggunakan looping `do-while`, kita membuat menu yang didalamnya berisi `switch`. Jika kita mengiputkan angka 1 maka program otomatis akan mengarahkan untuk mencari nilai maksimum, angka 2 akan mengarahkan untuk mencari nilai minimum, angka ke 3 akan mengarahkan untuk mencari rata-rata. Lalu untuk mengakhiri menu atau looping, kita bisa menginputkan angka 0. Jika kita menginputkan angka selain 1, 2, 3, ataupun 0 maka otomatis akan muncul pesan error. Jika user menginputkan angka selain 0 maka program otomatis akan mengulang hingga user menginputkan angka 0 untuk mengakhiri loop.

#### Full code Screenshot:
![fullcode3](https://github.com/agungibr/Struktur-Data-Assignment/assets/91455543/d6a6229f-6bce-4b2e-9a1a-7164042c6d3a)


## Kesimpulan
Pada modul 2 guided ini kita belajar mengenai apa itu array hingga jenis-jenis array dan implementasinya. Array dibagi menjadi 3 sesuai dengan jenisnya, yaitu array satu dimensi yang merupakan kumpulan elemen yang bersebelahan di mana individu elemen diidentifikasi oleh subskrip integer unik yang dimulai dengan nol, lalu array dua dimensi yaitu suatu kumpulan elemen yang dapat dilakukan penyimpanan secara terurut di suatu tempat yang terbagi menjadi dua 2 unsur yakni baris dan kolom, dan yang terakhir ada array multidimensi yang digunakan untuk digunakan untuk merepresentasikan array dengan dimensi lebih dari dua atau array yang memiliki lebih dari dua indeks, seperti array tiga dimensi, array empat dimensi, array lima dimensi, dan seterusnya. 

Lalu untuk modul 2 unguided kita belajar bagaimana cara implementasi dari array tiga dimensi menggunakan fungsi input, memisahkan atau memfilter nilai array berdasarkan dengan angka ganjil atau angka genap, dan membuat program menu untuk mencari nilai maksimum, minimum, dan rata-rata dari suatu array dengan menggunakan fungsi inputan.

## Referensi
[1] Mulyana, A., et al., Cara Mudah Mempelajari Algoritma dan Struktur Data. Edisi Pertama. Yogyakarta : DIVA Press, 2021

[2] Zein, A. dan Eriana, E.S., Algoritma dan Struktur Data. Edisi Pertama. Tangerang Selatan : Unpam Press, 2022.

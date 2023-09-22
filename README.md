# Ψηφιακή Επεξεργασία Εικόνας

Θέμα εργασίας: “Κατηγοριοποίηση Εικόνων – Σύγκριση δύο αντιπροσωπευτικών μεθόδων”

## Εγκατάσταση

Για την εγκατάσταση και εκτέλεση του κώδικα απαιτούνται
Python3 (3.5+) και git (για το κατέβασμα).

Οι οδηγίες που παρέχονται είναι για Linux / Mac. Για Windows τα ``pip3``
και ``python3`` θα πρέπει να αλλάξουν σε ``pip`` και ``python``.

### Βήματα

1. Κατέβασμα του κώδικα

```sh
git clone https://github.com/mdrosiadis/project-psifiaki-epexergasia-eikonas
cd project-psifiaki-epexergasia-eikonas
```

2. Εγκατάσταση προαπαιτούμενων

```sh
pip3 install -r requirements.txt
```

3. Εκτέλεση του κώδικα

```sh
python3 src/cnn.py
python3 src/hog.py
```

Μετά την εκτέλεση του κώδικα, παράγονται αρχεία αποθήκευσης των μοντέλων,
ώστε να μην υπολογίζονται εκ νέου κάθε φορά. Για την εκαθάριση αυτών
των αρχείων, εκτελούμε:
```sh
rm model_cnn.data model_hog.data data/hog_dataset.gz
```

Τα γραφήματα με τα αποτελέσματα παράγονται στον φάκελο ***plots***.

Στο φάκελο ***report*** βρίσκεται η αναφορά της εργασίας.

Στο αρχείο ***results.txt*** βρίσκεται ενδεικτική έξοδος
διάφορων εκτελέσεων των αλγορίθμων.

## Introduction
Точное предсказание энергетических характеристик молекул имеет важное значение в различных областях, включая квантовую химию, катализ, материаловедение и разработку лекарственных препаратов. Традиционные методы, основанные на ab-initio вычислениях, могут быть очень затратными по времени и вычислительным ресурсам, особенно для больших молекулярных систем.

В этом проекте я использую [MatErials Graph Network (MEGNet)](https://github.com/materialsvirtuallab/megnet), одну из самых популярных популярных графовых нейронных сетей для предсказания свойств молекул по их графовому представлению. Ключевая особенность [MEGNet](https://github.com/materialsvirtuallab/megnet) в сравнении с другими моделями заключается в ее легковестности и в передовом использовании тройных типов фичей для каждой молекулы, которые стали основоположниками в последующих исследованиях и разработках, например, [M3GNet](https://www.nature.com/articles/s43588-022-00349-3).

Задача состояла в создании модели, чья MAE не превышала бы 1 *meV*, т.е. погрешность DFT для рассчета энергии молекул. Для этого тестировали гипотезы по взаимодействию с молекулами.

### Solution
Анализ датасета и создание baseline - [analisys](https://github.com/Gruz2520/predict_energy_of_mols/blob/main/notebooks/analisys.ipynb)

Обучение модели и валидация - [train](https://github.com/Gruz2520/predict_energy_of_mols/blob/main/notebooks/training.ipynb)

Обучение на датасете с добавлением сэмплов из qm9 - [train_qm9](https://github.com/Gruz2520/predict_energy_of_mols/blob/main/notebooks/train_with_qm9.ipynb)

В папке [other models](https://github.com/Gruz2520/predict_energy_of_mols/tree/main/other_models) тестировали другие модели для последующего сравнительного анализа результатов.

### Results
|Model|MAE|
|-----|---|
|MEGNet|0.0017 meV\atom|
|schenet|0.0042 meV\atom|
|dimenet++|0.0066 meV\atom|
|comenet|0.0031 meV\atom|

Результаты вычислений были произведены на кластере ВШЭ. 

### Requirements

Из-за того, что основной репазиторий перестал поддерживаться, а основные библиотеки скакнули вперед, то пришлось переписать некоторые исходники, чтобы она нормально запускалась.

- Python 3.7.9

- local_env.py file for pymatgen

    path for change: ../pymatgen/analysis/local_env.py

- callbacks.py for megnet

    path for change: ../megnet/callbacks.py

```bash
pip install -r req.txt
```
